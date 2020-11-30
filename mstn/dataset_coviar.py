"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""


import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath) # 把项目的根目录添加到程序执行时的环境变量

import torch
import torch.utils.data as data
import torchvision.models.resnet
import torchvision
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import os.path
import random
import numpy as np
from coviar import get_num_frames
from coviar import load
from transforms import *
from utils.sample import random_sample, fix_sample

GOP_SIZE = 12

IFRAME = 0
MV = 1
RESIDUAL = 2

num = 0
def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


# def get_seg_range(n, num_segments, seg, representation):
#     if representation in ['residual', 'mv']:
#         n -= 1
#     # if n < num_segments:
#         # print("-------------------------------- n < samples ")
#     seg_size = float(n - 1) / num_segments
#     seg_begin = int(np.round(seg_size * seg))
#     seg_end = int(np.round(seg_size * (seg + 1)))
#     if seg_end == seg_begin:
#         seg_end = seg_begin + 1
#
#     if representation in ['residual', 'mv']:
#         # Exclude the 0-th frame, because it's an I-frame.
#         return seg_begin + 1, seg_end + 1
#     assert seg_end < n,print('out of bound when sampling')
#     return seg_begin, seg_end

def get_seg_range(n, num_segments, seg, representation):
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        # Exclude the 0-th frame, because it's an I-frmae.
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end

def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root,dataset,
                 video_list,
                 num_segments,
                 alpha,
                 is_train):

        self._data_root = data_root
        self._dataset = dataset
        self._num_segments = num_segments
        self._mv_scales = [1, .875, .75]
        self._iframe_scales = [1, .875, .75, .66]
        self._input_size = 224
        self._scale_size = self._input_size * 256 // 224
        self._iframe_transform = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, self._iframe_scales),
             GroupRandomHorizontalFlip(is_mv=False)])
        self._mv_transform = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, self._mv_scales),
             GroupRandomHorizontalFlip(is_mv=True)])
        self._infer_transform = torchvision.transforms.Compose([
            GroupScale(int(self._scale_size)),
            GroupCenterCrop(self._input_size),
        ])
        self.alpha = alpha
        self._is_train = is_train
        self._accumulate = True
        ## use kinetic pretrain setting
        self._input_mean = np.array([0.43216, 0.394666, 0.37645]).reshape((3, 1, 1, 1))
        self._input_std = np.array([0.22803, 0.22145, 0.216989]).reshape((3, 1, 1, 1))

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        if self._dataset == 'ucf101': # for ucf
            with open(video_list, 'r') as f:
                for line in f:
                    video,_,label = line.strip().split(' ')
                    video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                    self._video_list.append((
                        video_path,
                        int(label),
                        get_num_frames(video_path)))
        if self._dataset == 'kinetics400':  # for kinetics
            with open(video_list, 'r') as f:
                for line in f:
                    video, label = line.strip().split(',')
                    video_path = os.path.join(self._data_root, video)
                    self._video_list.append((
                        video_path,
                        int(label),
                        get_num_frames(video_path)))

        print('%d videos loaded.' % len(self._video_list))

    def _get_train_frame_index(self, num_frames, seg, num_segments,representation):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, num_segments, seg, representation)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, representation)

    def _get_test_frame_index(self, num_frames, seg,num_segments, representation):
        if representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, representation)

    def __getitem__(self, index):

        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        iframes = []
        mvs = []
        for seg in range(self._num_segments):

            # for iframe
            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg, self._num_segments, 'iframe')
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg,self._num_segments, 'iframe')

            img = load(video_path, gop_index, gop_pos, IFRAME, self._accumulate)
            if img is None:
                print('Error: loading video %s failed.' % video_path)
                img = np.zeros((224, 224, 3))
            # img = color_aug(img)
            # BGR to RGB. (PyTorch uses RGB according to doc.)
            img = img[..., ::-1]
            iframes.append(img)

        for seg in range(self._num_segments * self.alpha):
            # for mv .notice here we should use the same gop_index with iframe
            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg,self._num_segments * self.alpha, 'mv')
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg,self._num_segments * self.alpha, 'mv')
            mv = load(video_path, gop_index, gop_pos, MV, self._accumulate)
            if mv is None:
                print('Error: loading video %s failed.' % video_path)
                mv = np.zeros((224, 224, 2))

            mv = clip_and_scale(mv, 20)  # scale up the value
            mv += 128
            mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)
            mvs.append(mv)

        # preprocess iframe
        iframes = self._iframe_transform(iframes) if self._is_train else self._infer_transform(iframes)
        iframes = np.asarray(iframes,dtype=np.float32) / 255.0
        iframes = np.transpose(iframes, (3, 0, 1, 2))
        iframes = (iframes - self._input_mean) / self._input_std


        # preprocess mv
        mvs = self._mv_transform(mvs) if self._is_train else self._infer_transform(mvs)
        mvs = np.asarray(mvs, dtype=np.float32) / 255.0
        mvs = np.transpose(mvs, (3, 0, 1, 2))
        mvs = (mvs - 0.5)

        # check the shape
        # channels,depth, width, height
        assert iframes.shape[1] == self._num_segments, print("iframe shape wrong")
        assert mvs.shape[1] == self._num_segments * self.alpha, print("timesacle shape wrong")
        return (iframes, mvs), label
        #todo check the mv shape size

    def __len__(self):
        return len(self._video_list)


def visualize_mv(mat):
    # Use Hue, Saturation, Value colour model
    mat = mat.astype(np.float32)
    w = mat.shape[0]
    h = mat.shape[1]
    hsv = np.zeros((w, h, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(mat[..., 0], mat[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.imshow(bgr_frame)
    plt.show()


if __name__ == '__main__':
    import time
    start = time.time()
    from config import Config
    cfg = Config()
    cfg.parse({'train_data_root': r'/home/sjhu/datasets/kinetics400_mpeg4/train_256',
               'test_data_root': r'/home/sjhu/datasets/kinetics400_mpeg4/val_256',
               'dataset': 'kinetics400',
               'model': 'model1_kinetics',
               'train_list': r'/home/sjhu/datasets/k400_train_sample.txt',
               'test_list': r'/home/sjhu/datasets/k400_debug.txt',
               'gpus': [0, 1, 2, 3],
               'batch_size': 32,
               'alpha': 2,
               'num_segments': 4,
               'workers': 32,
               })
    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            cfg.test_data_root,
            dataset= cfg.dataset,
            video_list=cfg.test_list,
            num_segments=cfg.num_segments,
            alpha=cfg.alpha,
            is_train=True,
        ),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, drop_last=False, pin_memory=True)


    for i, (input_pairs, label) in enumerate(train_loader):
        iframe, mv= input_pairs
        print(iframe.shape)
        print(mv.shape)
        print("%d" % i)
    end = time.time()
    print("cost %f s" % ((end - start)))