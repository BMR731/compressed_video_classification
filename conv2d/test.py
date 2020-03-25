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
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['acc_max']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    tfc = TransformsConfig()
    # if args.test_crops == 1:
    #     cropping = torchvision.transforms.Compose([
    #         GroupScale(tfc.scale_size),
    #         GroupCenterCrop(tfc.crop_size),
    #     ])
    # elif args.test_crops == 10:
    #     cropping = torchvision.transforms.Compose([
    #         GroupOverSample(tfc.crop_size, tfc.scale_size, is_mv=(args.representation == 'mv'))
    #     ])
    # else:
    #     raise ValueError("Only 1 and 10 crops are supported, but got {}.".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.test_list,
            num_segments=args.test_segments,
            rgb_transforms=torchvision.transforms.Compose([
                GroupOverSample(tfc.crop_size, tfc.scale_size, is_mv=False)
            ]),
            mv_transforms=torchvision.transforms.Compose([
                GroupOverSample(tfc.crop_size, tfc.scale_size, is_mv=True)
            ]),
            is_train=False,
            accumulate=True,
        ),
        batch_size=1, shuffle=False,
        num_workers=args.workers*2, pin_memory=True)

    devices = [torch.device("cuda:%d" % device) for device in args.gpus]
    net = torch.nn.DataParallel(net, device_ids=args.gpus)
    net =net.to(devices[0])
    net.eval()


    output = []
    total_num = len(data_loader.dataset)
    proc_start_time = time.time()
    for i, (input_pairs, label) in enumerate(data_loader):
        with torch.no_grad():
            input_pairs[0] = input_pairs[0].float().to(devices[0])
            input_pairs[1] = input_pairs[1].float().to(devices[0])
            iscores, _, _, _ = net(input_pairs)
            output.append((iscores.cpu().numpy(),label[0]))
            cnt_time = time.time() - proc_start_time
            if (i + 1) % 100 == 0:
                print('video {} done, total {}/{}, average {} sec/video'.format(i, i + 1,
                                                                                total_num,
                                                                                float(cnt_time) / (i + 1)))

    video_pred = [np.argmax(x[0]) for x in output]
    video_labels = [x[1] for x in output]

    print('Accuracy {:.02f}% ({})'.format(
        float(np.sum(np.array(video_pred) == np.array(video_labels))) / len(video_pred) * 100.0,
        len(video_pred)))


if __name__ == '__main__':
    main()
