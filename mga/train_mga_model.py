"""Run training."""

import shutil
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mga.dataset_mga_model import CoviarDataSet
from mga.mga_model import IframeNet
from train_options import parser

from torch.utils.tensorboard import SummaryWriter

SAVE_FREQ = 5
PRINT_FREQ = 20
ACCUMU_STEPS = 2  # use gradient accumlation to use least memory and more runtime
loss_min = 1
CONTINUE_FROM_LAST = False
LAST_SAVE_PATH = r'r2plus1d_18_bt_24_seg_10_ifame_fusion_qp+mv_part_dataset_checkpoint.pth.tar'
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
    description = 'bt_%d_seg_%d_wd_%f_%s' % (args.batch_size*2, args.num_segments,args.weight_decay, "train_joint")
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
        print("model epoch {} lowest loss {}".format(checkpoint['epoch'], checkpoint['loss_min']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        loss_min = checkpoint['loss_min']
        model.load_state_dict(base_dict)
        start_epochs = checkpoint['epoch']
    else:
        loss_min = 10000
        start_epochs = 0

    # print(model)
    # WRITER.add_graph(model, (torch.randn(10,5, 2, 224, 224),))

    devices = [torch.device("cuda:%d" % device) for device in args.gpus]
    global DEVICES
    DEVICES = devices

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.train_list,
            num_segments=args.num_segments,
            representation=args.representation,
            is_train=True,
            accumulate=(not args.no_accumulation),
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.test_list,
            num_segments=args.num_segments,
            representation=args.representation,
            is_train=False,
            accumulate=(not args.no_accumulation),
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model = model.to(devices[0])
    cudnn.benchmark = True

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0
        if '.fc.' in key:
            lr_mult = 1.0
            lr = args.lr
        elif 'conv1x1' in key:
            lr_mult = 0.1
            lr = args.lr*10
        else:
            lr_mult = 0.01
            lr = args.lr

        params += [{'params': value, 'lr': lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]

    if FINETUNE:
        optimizer = torch.optim.SGD(params, lr=1e-5, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(
            params,
            weight_decay=args.weight_decay,
            eps=0.001)

    criterions = torch.nn.CrossEntropyLoss().to(devices[0])
    # criterions = LabelSmoothingLoss(101,0.1,-1)

    # try to use ReduceOnPlatue to adjust lr
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20 // args.eval_freq, verbose=True)

    for epoch in range(start_epochs, args.epochs):
        # about optimizer
        WRITER.add_scalar('Lr/epoch', get_lr(optimizer), epoch)
        loss_train = train(train_loader, model, criterions, optimizer, epoch)
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss_val, acc = validate(val_loader, model, criterions, epoch)
            scheduler.step(loss_val)
            is_best = (loss_val < loss_min)
            loss_min = min(loss_val, loss_min)
            # visualization
            WRITER.add_scalar('Accuracy/epoch', acc, epoch)
            WRITER.add_scalars('Classification Loss/epoch', {'Train': loss_train, 'Val': loss_val}, epoch)
            if is_best or epoch % SAVE_FREQ == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'loss_min': loss_min,
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
    clf_losses = AverageMeter()

    model.train()
    end = time.time()
    for i, (input_pairs, label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_pairs[0] = input_pairs[0].float().to(devices[0])
        input_pairs[1] = input_pairs[1].float().to(devices[0])
        label = label.float().to(devices[0])
        outputs,img,mga = model(input_pairs)
        loss = criterions(outputs, label.clone().long()) / ACCUMU_STEPS
        clf_losses.update(loss.item(), args.batch_size)
        loss.backward(retain_graph=True)
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
                   'classifier Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'.format(
                epoch, i, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss2=clf_losses)))

    return clf_losses.avg  # attention indent ,there was a serious bug here


def validate(val_loader, model, criterions, epoch):
    '''
    :param val_loader:
    :param model:
    :param criterions:
    :param epoch:
    :return:  (clf loss, acc)
    '''
    batch_time = AverageMeter()
    clf_losses = AverageMeter()
    model.eval()
    end = time.time()
    correct_nums = 0
    for i, (input_pairs, label) in enumerate(val_loader):
        with torch.no_grad():
            input_pairs[0] = input_pairs[0].float().to(devices[0])
            input_pairs[1] = input_pairs[1].float().to(devices[0])
            label = label.float().to(devices[0])

            y,_,_ = model(input_pairs)
            loss = criterions(y, label.clone().long())
            clf_losses.update(loss.item(), args.batch_size)

            _, predicts = torch.max(y, 1)
            correct_nums += (predicts == label.clone().long()).sum()
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print(('Validate: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'clf loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                    .format(
                    i, len(val_loader),
                    batch_time=batch_time,
                    loss2=clf_losses)))

    acc = float(100 * correct_nums) / len(val_loader.dataset)
    print((
        'Validating Results: classification loss {loss3.avg:.5f}, Accuracy: {accuracy:.3f}%'.format(loss3=clf_losses,
                                                                                                    accuracy=acc)))
    return clf_losses.avg, acc


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((description, filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((description, '_best.pth.tar'))
        shutil.copyfile(filename, best_name)


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr


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
