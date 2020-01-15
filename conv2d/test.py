"""Run testing given a trained model."""

import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision

from conv2d.dataset import CoviarDataSet
from conv2d.model import IframeNet
from transforms import *

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--data-root', type=str)
parser.add_argument('--test-list', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--test-segments', type=int, default=25)
parser.add_argument('--test-crops', type=int, default=10)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of workers for data loader.')
parser.add_argument('--gpus', nargs='+', type=int, default=None)

args = parser.parse_args()


def main():
    net = IframeNet(args.test_segments * args.test_crops)
    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.crop_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.crop_size, net.scale_size, is_mv=(args.representation == 'mv'))
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported, but got {}.".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            num_segments=args.test_segments,
            rgb_transforms=cropping,
            mv_transforms=cropping,
            is_train=False,
            accumulate=True,
        ),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.to(devices[0]), device_ids=devices)
    net.eval()

    correct_nums = 0
    proc_start_time = time.time()
    for i, (input_pairs, label) in enumerate(data_loader):
        input_pairs[0] = input_pairs[0].float().to(devices[0])
        input_pairs[1] = input_pairs[1].float().to(devices[0])
        label = label.float().to(devices[0])
        iscores, _, _, _ = net(input_pairs)
        _, predicts = torch.max(iscores, 1)
        correct_nums += (predicts == label.clone().long()).sum()

    acc = float(100 * correct_nums) / len(data_loader.dataset)
    print("Accuracy: %.2f%%" % acc)
    print("Time: %f" % (time.time() - proc_start_time))


if __name__ == '__main__':
    main()
