"""Run training."""

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
# for terminal run
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(cur_dir) + os.path.sep + ".")
sys.path.append(root_path)  # 把项目的根目录添加到程序执行时的环境变量
# # #

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
import shutil
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.lr_scheduler import WarmStartCosineAnnealingLR
from mtsa.dataset_gjy import CoviarDataSet
from mtsa.model_modified import SlowFast
from train_options import parser
from torch.utils.tensorboard import SummaryWriter



SAVE_FREQ = 5
PRINT_FREQ = 20
ACCUMU_STEPS = 4  # use gradient accumlation to use least memory and more runtime
loss_min = 1
CONTINUE_FROM_LAST = False
LAST_SAVE_PATH = r'/home/sjhu/projects/pytorch-coviar/mtsa/bt_64_seg_5_mtsa_checkpoint.pth.tar'
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
    description = 'bt_%d_seg_%d_%s' % (
        args.batch_size * ACCUMU_STEPS, args.num_segments, "mtsa")
    log_name = r'/home/sjhu/projects/pytorch-coviar/mtsa/log/%s' % description
    WRITER = SummaryWriter(log_name)
    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    model = SlowFast(class_num=101)

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

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.train_list,
            num_segments=args.num_segments,
            is_train=True,
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers,drop_last=False, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.test_list,
            num_segments=args.num_segments,
            is_train=False,
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, drop_last=False,pin_memory=True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model = model.to(devices[0])
    cudnn.benchmark = True

    # params_dict = dict(model.named_parameters())
    # params = []
    # for key, value in params_dict.items():
    #     decay_mult = 0.0 if 'bias' in key else 1.0
    #     if 'fc' in key:
    #         params += [{'params': [value], 'lr': args.lr, 'decay_mult': decay_mult}]
    #     elif 'conv1x1' in key:
    #         params += [{'params': [value], 'lr': args.lr, 'decay_mult': decay_mult}]
    #     elif 'module.mvnet' in key:
    #         params += [{'params': [value], 'lr': args.lr, 'decay_mult': decay_mult}]
    #     else:
    #         params += [{'params': [value], 'lr': args.lr, 'decay_mult': decay_mult}]

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, eps=0.001)
    scheduler = WarmStartCosineAnnealingLR(optimizer, args.epochs, 10)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=20 // args.eval_freq, verbose=True)
    criterions = []
    criterions.append(torch.nn.CrossEntropyLoss().to(devices[0]))
    # criterions = LabelSmoothingLoss(101,0.1,-1)

    # try to use ReduceOnPlatue to adjust lr

    for epoch in range(start_epochs, args.epochs):
        # about optimizer
        WRITER.add_scalar('Lr/epoch', get_lr(optimizer), epoch)
        loss_train = train(train_loader, model, criterions, optimizer, epoch)
        scheduler.step(epoch)
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss_val, acc = validate(val_loader, model, criterions, epoch)
            is_best = (acc > acc_max)
            acc_max = max(acc, acc_max)
            # visualization
            WRITER.add_scalar('Accuracy/epoch', acc, epoch)
            WRITER.add_scalars('Clf Loss/epoch', {'Train': loss_train, 'Val': loss_val}, epoch)

            if is_best or epoch % SAVE_FREQ == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
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
    losses = AverageMeter()

    model.train()
    end = time.time()
    for i, (input_pairs, label) in enumerate(train_loader):
        assert not np.isnan(np.min(input_pairs[0].numpy())), print("data has nan")
        assert not np.isnan(np.min(input_pairs[1].numpy())), print("data has nan")
        # assert not np.isnan(np.min(input_pairs[1].numpy())),print("data has nan")
        data_time.update(time.time() - end)
        input_pairs[0] = input_pairs[0].float().to(devices[0])
        input_pairs[1] = input_pairs[1].float().to(devices[0])
        input_pairs[2] = input_pairs[2].float().to(devices[0])
        input_pairs[3] = input_pairs[3].float().to(devices[0])

        label = label.float().to(devices[0])
        scores = model(input_pairs)
        loss = criterions[0](scores, label.clone().long()) / ACCUMU_STEPS

        losses.update(loss.item(), args.batch_size)
        loss.backward()

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
                loss2=losses)))

    return losses.avg  # attention indent ,there was a serious bug here


def validate(val_loader, model, criterions, epoch):
    '''
    :param val_loader:
    :param model:
    :param criterions:
    :param epoch:
    :return:  (clf loss, acc)
    '''
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    end = time.time()
    correct_nums = 0
    for i, (input_pairs, label) in enumerate(val_loader):
        assert not np.isnan(np.min(input_pairs[0].numpy())), print("data has nan")
        assert not np.isnan(np.min(input_pairs[1].numpy())), print("data has nan")
        with torch.no_grad():
            input_pairs[0] = input_pairs[0].float().to(devices[0])
            input_pairs[1] = input_pairs[1].float().to(devices[0])
            input_pairs[2] = input_pairs[2].float().to(devices[0])
            input_pairs[3] = input_pairs[3].float().to(devices[0])
            label = label.float().to(devices[0])

            scores = model(input_pairs)
            iloss = criterions[0](scores, label.clone().long()) / ACCUMU_STEPS

            losses.update(iloss.item(), args.batch_size)

            _, predicts = torch.max(scores, 1)
            correct_nums += (predicts == label.clone().long()).sum()
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print(('Validate: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Classifier Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                    .format(
                    i, len(val_loader),
                    batch_time=batch_time,
                    loss2=losses)))

    acc = float(100 * correct_nums) / len(val_loader.dataset)
    print((
        'Validating Results: clf loss {loss1.avg:.5f}, Accuracy: {accuracy:.3f}%'
            .format(loss1=losses, accuracy=acc)))
    return losses.avg, acc


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((description, filename))
    filename = r'/home/sjhu/projects/pytorch-coviar/mtsa/' + filename
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((description, '_best.pth.tar'))
        best_name = r'/home/sjhu/projects/pytorch-coviar/mtsa/' + best_name
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


if __name__ == '__main__':
    main()

