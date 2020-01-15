"""Run training."""

import shutil
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from conv2d.dataset import CoviarDataSet
from conv2d.model import IframeNet
from train_options import parser
from transforms import *
from torch.utils.tensorboard import SummaryWriter
from utils.lr_scheduler import WarmStartCosineAnnealingLR
from utils.label_smoothing import LabelSmoothingLoss
import torchvision

SAVE_FREQ = 5
PRINT_FREQ = 20
ACCUMU_STEPS = 4  # use gradient accumlation to use least memory and more runtime
loss_min = 1
CONTINUE_FROM_LAST = False
LAST_SAVE_PATH = r'bt_80_seg_3_wd_0.000100_only_use_mgc_layer34__best.pth.tar'
FINETUNE = False

# for visualization
WRITER = []
DEVICES = []

description = ""


def main():
    print(torch.cuda.device_count())
    global args
    global devices
    global WRITER
    args = parser.parse_args()
    global description
    description = 'bt_%d_seg_%d_wd_%f_%s' % (
        args.batch_size * ACCUMU_STEPS, args.num_segments, args.weight_decay, "layer234_ls_sgd")
    log_name = './log/%s' % description
    WRITER = SummaryWriter(log_name)
    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    model = IframeNet(args.num_segments)

    # add continue train from before
    if CONTINUE_FROM_LAST:
        checkpoint = torch.load(LAST_SAVE_PATH)
        # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        print("model epoch {} max acc {}".format(checkpoint['epoch'], checkpoint['acc_max']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        acc_max = checkpoint['acc_max']
        model.load_state_dict(base_dict)
        start_epochs = checkpoint['epoch']
    else:
        acc_max = -1
        start_epochs = 0

    devices = [torch.device("cuda:%d" % device) for device in args.gpus]
    global DEVICES
    DEVICES = devices

    tfc = TransformsConfig()
    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.train_list,
            num_segments=args.num_segments,
            is_train=True,
            accumulate=True,
            rgb_transforms=torchvision.transforms.Compose(
                [GroupMultiScaleCrop(tfc.input_size, tfc.iframe_scales),
                 GroupRandomHorizontalFlip(is_mv=False)]),
            mv_transforms=torchvision.transforms.Compose(
                [GroupMultiScaleCrop(tfc.input_size, tfc.mv_scales),
                 GroupRandomHorizontalFlip(is_mv=True)]),
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.test_list,
            num_segments=args.num_segments,
            is_train=False,
            accumulate=True,
            rgb_transforms=torchvision.transforms.Compose([
                GroupScale(int(tfc.scale_size)),
                GroupCenterCrop(tfc.input_size)]),
            mv_transforms=torchvision.transforms.Compose([
                GroupScale(int(tfc.scale_size)),
                GroupCenterCrop(tfc.input_size)]),
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model = model.to(devices[0])
    cudnn.benchmark = True

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0
        if 'module.fc' in key:
            params += [{'params': [value], 'lr': args.lr * 10, 'decay_mult': decay_mult}]
        elif 'conv1x1' in key:
            params += [{'params': [value], 'lr': args.lr * 10, 'decay_mult': decay_mult}]
        elif 'module.mvnet' in key:
            params += [{'params': [value], 'lr': args.lr * 10, 'decay_mult': decay_mult}]
        else:
            params += [{'params': [value], 'lr': args.lr * 1, 'decay_mult': decay_mult}]

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = WarmStartCosineAnnealingLR(optimizer, args.epochs, 10)
    # optimizer = torch.optim.Adam(params, weight_decay=args.weight_decay, eps=0.001)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=20 // args.eval_freq, verbose=True)
    criterions = []
    # criterions.append(torch.nn.CrossEntropyLoss().to(devices[0]))
    criterions.append(LabelSmoothingLoss(101, 0.1))
    criterions.append(torch.nn.KLDivLoss(reduction='batchmean').to(devices[0]))
    # criterions = LabelSmoothingLoss(101,0.1,-1)

    # try to use ReduceOnPlatue to adjust lr

    for epoch in range(start_epochs, args.epochs):
        # about optimizer
        WRITER.add_scalar('Lr/epoch', get_lr(optimizer), epoch)
        iloss_train, mloss_train, klloss_train = train(train_loader, model, criterions, optimizer, epoch)
        scheduler.step(epoch)
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            iloss_val, mloss_val, klloss_val, acc = validate(val_loader, model, criterions, epoch)
            # scheduler.step(acc)
            is_best = (acc > acc_max)
            acc_max = max(acc, acc_max)
            # visualization
            WRITER.add_scalar('IframeNet Accuracy/epoch', acc, epoch)
            WRITER.add_scalars('Iframe clf Loss/epoch', {'Train': iloss_train, 'Val': iloss_val}, epoch)
            WRITER.add_scalars('Mv clf Loss/epoch', {'Train': mloss_train, 'Val': mloss_val}, epoch)
            WRITER.add_scalars('KL Loss/epoch', {'Train': klloss_train, 'Val': klloss_val}, epoch)
            if is_best or epoch % SAVE_FREQ == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'acc_max': acc_max,
                    },
                    is_best,
                    filename='checkpoint.pth.tar')
    WRITER.close()


def train(train_loader, model, criterions, optimizer, epoch):
    '''
    :param train_loader:
    :param model:
    :param criterions:
    :param optimizer:
    :param epoch:
    :return:  (clf loss)
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    iframe_losses = AverageMeter()
    mv_losses = AverageMeter()
    kl_losses = AverageMeter()

    model.train()
    end = time.time()
    for i, (input_pairs, label) in enumerate(train_loader):
        assert not np.isnan(np.min(input_pairs[0].numpy())), print("data has nan")
        assert not np.isnan(np.min(input_pairs[1].numpy())), print("data has nan")
        # assert not np.isnan(np.min(input_pairs[1].numpy())),print("data has nan")
        data_time.update(time.time() - end)
        input_pairs[0] = input_pairs[0].float().to(devices[0])
        input_pairs[1] = input_pairs[1].float().to(devices[0])
        label = label.float().to(devices[0])
        iscores, mscores, img, img_aug = model(input_pairs)
        iloss = criterions[0](iscores, label.clone().long()) / ACCUMU_STEPS
        mloss = criterions[0](mscores, label.clone().long()) / ACCUMU_STEPS
        klloss = criterions[1](F.log_softmax(iscores, dim=1), F.softmax(mscores, dim=1)) / ACCUMU_STEPS

        iframe_losses.update(iloss.item(), args.batch_size)
        mv_losses.update(mloss.item(), args.batch_size)
        kl_losses.update(klloss.item(), args.batch_size)

        iloss.backward(retain_graph=True)
        mloss.backward()
        # klloss.backward()
        # use gradient accumulation
        if i % ACCUMU_STEPS == 0:
            # attention the following line can't be transplaced
            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}],\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'iframe classifier Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                   'mv classifier Loss {loss3.val:.4f} ({loss3.avg:.4f})\t'
                   'kl Loss {loss4.val:.4f} ({loss4.avg:.4f})\t'.format(
                epoch, i, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss2=iframe_losses, loss3=mv_losses, loss4=kl_losses)))

    return iframe_losses.avg, mv_losses.avg, kl_losses.avg  # attention indent ,there was a serious bug here


def validate(val_loader, model, criterions, epoch):
    '''
    :param val_loader:
    :param model:
    :param criterions:
    :param epoch:
    :return:  (clf loss, acc)
    '''
    batch_time = AverageMeter()
    iframe_losses = AverageMeter()
    mv_losses = AverageMeter()
    kl_losses = AverageMeter()
    model.eval()
    end = time.time()
    correct_nums = 0
    for i, (input_pairs, label) in enumerate(val_loader):
        assert not np.isnan(np.min(input_pairs[0].numpy())), print("data has nan")
        assert not np.isnan(np.min(input_pairs[1].numpy())), print("data has nan")
        with torch.no_grad():
            input_pairs[0] = input_pairs[0].float().to(devices[0])
            input_pairs[1] = input_pairs[1].float().to(devices[0])
            label = label.float().to(devices[0])

            iscores, mscores, img, img_aug = model(input_pairs)
            iloss = criterions[0](iscores, label.clone().long()) / ACCUMU_STEPS
            mloss = criterions[0](mscores, label.clone().long()) / ACCUMU_STEPS
            klloss = criterions[1](F.log_softmax(iscores, dim=1), F.softmax(mscores, dim=1)) / ACCUMU_STEPS

            iframe_losses.update(iloss.item(), args.batch_size)
            mv_losses.update(mloss.item(), args.batch_size)
            kl_losses.update(klloss.item(), args.batch_size)

            _, predicts = torch.max(iscores, 1)
            correct_nums += (predicts == label.clone().long()).sum()
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print(('Validate: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'iframe classifier Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                       'mv classifier Loss {loss3.val:.4f} ({loss3.avg:.4f})\t'
                       'kl Loss {loss4.val:.4f} ({loss4.avg:.4f})\t'
                    .format(
                    i, len(val_loader),
                    batch_time=batch_time,
                    loss2=iframe_losses, loss3=mv_losses, loss4=kl_losses)))

    acc = float(100 * correct_nums) / len(val_loader.dataset)
    print((
        'Validating Results: iframe clf loss {loss1.avg:.5f},mv clf loss {loss2.avg:.5f},'
        'kl loss {loss3.avg:.5f}, IframeNet Accuracy: {accuracy:.3f}%'
            .format(loss1=iframe_losses, loss2=mv_losses, loss3=kl_losses, accuracy=acc)))
    return iframe_losses.avg, mv_losses.avg, kl_losses.avg, acc


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((description, filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((description, '_best.pth.tar'))
        shutil.copyfile(filename, best_name)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == '__main__':
    main()
