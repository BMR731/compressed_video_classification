"""Run training."""

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
# for terminal run
import os
import sys
import gc

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(cur_dir) + os.path.sep + ".")
sys.path.append(root_path)  # 把项目的根目录添加到程序执行时的环境变量
# # #
print('current dir:')
print(cur_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import shutil
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mstn.dataset_coviar import CoviarDataSet
from mstn.model3 import MSTN
from torch.utils.tensorboard import SummaryWriter
from utils.lr_scheduler import WarmStartCosineAnnealingLR, get_lr
from utils.label_smoothing import LabelSmoothingLoss
from config import Config
from utils.metric import performance_detail

cfg = Config()
# cfg.parse({'train_data_root': r'/home/sjhu/datasets/UCF-101-mpeg4',
#             'test_data_root': r'/home/sjhu/datasets/UCF-101-mpeg4',
#            'dataset': 'ucf101',
#             'model': 'model1',
#            'train_list': r'/home/sjhu/datasets/ucf101_split1_train.txt',
#            'test_list' : r'/home/sjhu/datasets/ucf101_split1_test.txt',
#            'gpus': [0,1,2,3],
#            'batch_size': 32,
#            'alpha':4,
#            'num_segments':4,
#            'workers':32,
#            })

cfg.parse({'train_data_root': r'/home/sjhu/datasets/kinetics400_mpeg4/val_256',
           'test_data_root': r'/home/sjhu/datasets/kinetics400_mpeg4/val_256',
           'dataset': 'kinetics400',
           'model': 'debug',
           'train_list': r'/home/sjhu/datasets/k400_debug.txt',
           'test_list': r'/home/sjhu/datasets/k400_debug.txt',
           'gpus': [0,1,2,3],
           'lr': 1e-3,
           'batch_size': 32,
           'alpha': 8,
           'num_segments': 4,
           'workers': 12,
           'weight_decay': 1e-4,
           'eval_freq': 5,
           'lr_steps': [40,40,50]
           })

SAVE_FREQ = 5
PRINT_FREQ = 20
ACCUMU_STEPS = 4  # use gradient accumlation to use least memory and more runtime
loss_min = 1
CONTINUE_FROM_LAST = True
# LAST_SAVE_PATH = r'/home/sjhu/projects/pytorch-coviar/mstn/bt16*4_seg_4_model3_k400_full_checkpoint.pth.tar'
LAST_SAVE_PATH = r'/home/sjhu/projects/pytorch-coviar/mstn/bt16*4_seg_4_model3_k400_full_best.pth.tar'
FINETUNE = False


def main():
    print(torch.cuda.device_count())
    global devices
    global WRITER
    global description
    description = 'bt%d*%d_seg_%d_%s' % (
        cfg.batch_size, ACCUMU_STEPS, cfg.num_segments, cfg.model)
    log_name = cur_dir + r'/log/%s' % description
    WRITER = SummaryWriter(log_name)
    # print('Training arguments:')
    # for k, v in vars(cfg).items():
    #     print('\t{}: {}'.format(k, v))

    if cfg.dataset == 'ucf101':
        num_classes = 101
    elif cfg.dataset == 'kinetics400':
        num_classes = 400

    model = MSTN(num_classes=num_classes, alpha=cfg.alpha)

    # add continue train from before
    if CONTINUE_FROM_LAST:
        checkpoint = torch.load(LAST_SAVE_PATH)
        acc_max = checkpoint['acc_max']
        start_epochs = checkpoint['epoch']
        print("model epoch {} max acc {}".format(checkpoint['epoch'], checkpoint['acc_max']))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict,strict=False)
    else:
        acc_max = -1
        start_epochs = 0

    devices = [torch.device("cuda:%d" % device) for device in cfg.gpus]
    global DEVICES
    DEVICES = devices

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            cfg.train_data_root,
            dataset=cfg.dataset,
            video_list=cfg.train_list,
            num_segments=cfg.num_segments,
            alpha=cfg.alpha,
            is_train=True,
        ),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, drop_last=False, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            cfg.test_data_root,
            dataset=cfg.dataset,
            video_list=cfg.test_list,
            num_segments=cfg.num_segments,
            alpha=cfg.alpha,
            is_train=False,
        ),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, drop_last=False, pin_memory=True)

    model = torch.nn.DataParallel(model, device_ids=cfg.gpus)
    model = model.to(devices[0])
    cudnn.benchmark = True

    # visualize(val_loader,model,WRITER)
    # return
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=cfg.weight_decay, eps=0.001)
    # scheduler = WarmStartCosineAnnealingLR(optimizer, cfg.epochs, 10)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5 // cfg.eval_freq, verbose=True)
    criterions = []
    criterions.append(torch.nn.CrossEntropyLoss().to(devices[0]))
    # criterions.append(LabelSmoothingLoss(101, 0.1, -1).to(devices[0]))

    for epoch in range(start_epochs, cfg.epochs):
        # about optimizer
        # cur_lr = adjust_learning_rate(optimizer, epoch, cfg.lr_steps, cfg.weight_decay)
        WRITER.add_scalar('Lr/epoch', get_lr(optimizer), epoch)
        loss_train, acc_train = train(train_loader, model, criterions, optimizer, epoch)
        if epoch % cfg.eval_freq == 0 or epoch == cfg.epochs - 1:
            loss_val, acc = validate(val_loader, model, criterions, epoch)
            is_best = (acc > acc_max)
            acc_max = max(acc, acc_max)
            scheduler.step(acc)
            # visualization
            WRITER.add_scalars('Accuracy/epoch', {'Train': acc_train, 'Val': acc}, epoch)
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
    correct_nums = 0
    for i, (input_pairs, label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_pairs[0] = input_pairs[0].float().to(devices[0])
        input_pairs[1] = input_pairs[1].float().to(devices[0])
        label = label.float().to(devices[0])
        score = model(input_pairs)
        loss = criterions[0](score, label.clone().long()) / ACCUMU_STEPS
        losses.update(loss.detach().item(), cfg.batch_size)
        loss.backward()

        _, predicts = torch.max(score, 1)
        correct_nums += (predicts == label.clone().long()).sum()

        # use gradient accumulation
        if i % ACCUMU_STEPS == 0:
            # attention the following line can't be transplaced
            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        acc = float(100 * correct_nums) / len(train_loader.dataset)
        if i % PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}],\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'classifier Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'.format(
                epoch, i, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss2=losses)))
        gc.collect()

    return losses.avg, acc  # attention indent ,there was a serious bug here


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

    scores = []
    y = []
    for i, (input_pairs, label) in enumerate(val_loader):
        with torch.no_grad():
            input_pairs[0] = input_pairs[0].float().to(devices[0])
            input_pairs[1] = input_pairs[1].float().to(devices[0])
            label = label.float().to(devices[0])

            score = model(input_pairs)
            loss = criterions[0](score, label.clone().long()) / ACCUMU_STEPS
            losses.update(loss.detach().item(), cfg.batch_size)

            _, predicts = torch.max(score, 1)
            correct_nums += (predicts == label.clone().long()).sum()
            batch_time.update(time.time() - end)
            end = time.time()

            scores.append(score.detach().cpu().numpy())
            y.append(label.detach().cpu().numpy())

            if i % PRINT_FREQ == 0:
                print(('Validate: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Classifier Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                    .format(
                    i, len(val_loader),
                    batch_time=batch_time,
                    loss2=losses)))

        gc.collect()
    performance_detail(y, scores)
    acc = float(100 * correct_nums) / len(val_loader.dataset)
    print((
        'Validating Results: clf loss {loss1.avg:.5f}, Accuracy: {accuracy:.3f}%'
            .format(loss1=losses, accuracy=acc)))
    return losses.avg, acc


def visualize(data_loader, net, writer):
    from torchvision.utils import make_grid
    for i, (input_pairs, label) in enumerate(data_loader):
        input_pairs[0] = input_pairs[0].float().to(devices[0])
        input_pairs[1] = input_pairs[1].float().to(devices[0])
        x1 = input_pairs[0][0]
        x2 = input_pairs[1][0]
        break
    net.eval()

    x1 = make_grid(x1, normalize=True, scale_each=True, nrow=8)
    x2 = make_grid(x2, normalize=True, scale_each=True, nrow=8)

    for name, module in net.named_children():
        print(name)
        print('---->')
        print(module)

    for name, layer in net._modules.items():
        # x2 = x2.view(x2.size(0), -1) if 'fc' in name else x2
        # print(x2.size())
        #
        # x2 = layer(x2)
        # print(f'{name}')
        #
        # # 查看卷积层的特征图
        # if 'layer' in name or 'conv' in name:
        #     x22 = x2.transpose(0, 1)  # C，B, H, W ---> B，C, H, W
        #     img_grid = make_grid(x22, normalize=True, scale_each=True, nrow=4)
        #     writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)
        print(name)


def save_checkpoint(state, is_best, filename):
    filename = cur_dir + '/' + '_'.join((description, filename))
    torch.save(state, filename)
    if is_best:
        best_name = cur_dir + '/' + '_'.join((description, 'best.pth.tar'))
        shutil.copyfile(filename, best_name)


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


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = cfg.lr * decay
    wd = cfg.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr

if __name__ == '__main__':
    main()
