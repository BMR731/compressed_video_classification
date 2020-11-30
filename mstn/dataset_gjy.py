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

import os
import os.path
import random
import math
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import traceback
import logging
import coviexinfo
import matplotlib.pyplot as plt
import torchvision
from utils.sample import *

from transforms import *

# from memory_profiler import profile
import gc
## For Dataset
WIDTH = 256
HEIGHT = 340

class CoviarDataSet(data.Dataset):
    def __init__(self, data_root,
                 video_list,
                 num_segments,
                 alpha,
                 is_train):

        self._data_root = data_root
        self._num_segments = num_segments
        self._alpha = alpha
        self._is_train = is_train
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
        # modify depend on Kinetics-400 dataset setting
        self._input_mean = np.array([0.43216, 0.394666, 0.37645]).reshape((3, 1, 1, 1)).astype(np.float32)
        self._input_std = np.array([0.22803, 0.22145, 0.216989]).reshape((3, 1, 1, 1)).astype(np.float32)

        self._load_list(video_list)


    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r') as f:
            for line in f:
                video,_,label = line.strip().split(' ') # for ucf
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')  #TODO for ucf101 config

                # video, label = line.strip().split(',') # for kinetics
                # video_path = os.path.join(self._data_root, video) # for k400 config
                self._video_list.append((
                    video_path,
                    int(label)))

        print('%d videos loaded.' % len(self._video_list))

    def __getitem__(self, index):
        # shapes  (Batch,C ,T,W,H)
        video, label = self._video_list[index]
        print(video)
        video_features = []
        extracter = VideoExtracter(video)

        # process mv
        mvs = extracter.load_mvs(self._num_segments*self._alpha, self._is_train)

        mvs = self._mv_transform(mvs) if self._is_train else self._infer_transform(mvs)
        mvs = np.asarray(mvs, dtype=np.float32) / 255.0
        mvs = np.transpose(mvs, (3, 0, 1, 2))
        mvs = (mvs - 0.5)

        # # process iframe
        iframes = extracter.load_keyframes(self._num_segments, self._is_train)
        iframes = self._iframe_transform(iframes) if self._is_train else self._infer_transform(iframes)
        iframes = np.asarray(iframes,dtype=np.float32) / 255.0
        iframes = np.transpose(iframes, (3, 0, 1, 2))
        iframes = (iframes - self._input_mean) / self._input_std

        assert iframes.shape[1] == self._num_segments, print("iframe shape wrong")
        assert mvs.shape[1] == self._num_segments*self._alpha, print("mv shape wrong")


        video_features.append(iframes)
        video_features.append(mvs)
        # print(iframes.shape)
        # print(mvs.shape)
        print('video extract success')

        return video_features, label


    def __len__(self):
        return len(self._video_list)


class VideoExtracter:
    def __init__(self, video_name):
        self.video_name = video_name
        # get basic decode information

    def load_keyframes(self, num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width, height, channels)
        """

        frames = coviexinfo.extract(self.video_name, 'get_I', 'train', 20, 0)
        if frames is None:
            print(self.video_name)
            print("frame,error this video has problems, is None")
            mat = np.random.randint(255, size=(num_segments, WIDTH, HEIGHT, 3))
            return np.array(mat, dtype=np.float32)


        mat = []
        for i in range(frames.shape[2]//3):
            rgb = np.dstack((frames[:, :, i * 3], frames[:, :, i * 3 + 1], frames[:, :, i * 3 + 2]))
            mat.append(color_aug(rgb))
            # plt.figure(i)
            # plt.subplot(2,1,1)
            # plt.imshow(rgb)
            # plt.subplot(2,1,2)
            # plt.imshow(rgb_a)
            # plt.show()
        mat = random_sample(mat, num_segments) if is_train else fix_sample(mat, num_segments)
        mat = np.asarray(mat, dtype=np.float32)
        return mat

    # @profile
    def load_mvs(self, num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width//4, height//4, channels=2) 0,255
        """
        # mv_ref_arr=(H/4,W/4,frames*6)
        # mv_ref_arr is a array with 3 dimensions. The first dimension denotes Height of a frame. The second dimension denotes Width of a frame.
        # For every frame, it contains mv_0_x, mv_0_y, ref_0, mv_1_x, mv_1_y, ref_1. So, the third dimension denote frames*6.
        phase = 'train' if is_train else 'test'
        mv_origin = coviexinfo.extract(self.video_name, 'get_mv', phase, num_segments, 0)
        if mv_origin is None:
            # print("mv, error this video has problems, is None")
            mat = np.full((num_segments, WIDTH//4, HEIGHT//4, 2),128)
            return np.array(mat, dtype=np.float32)



        mv_origin = mv_origin.reshape((mv_origin.shape[1],mv_origin.shape[2],mv_origin.shape[0]))
        mat = []
        mv_0_x = mv_origin[:, :, ::6]
        mv_0_y = mv_origin[:, :, 1::6]
        for i in range(mv_0_x.shape[2]):
            mv_0 = np.dstack((mv_0_x[:, :, i], mv_0_y[:, :, i]))
            mv_0 = clip_and_scale(mv_0, 20)
            mv_0 += 128
            mv_0 = (np.minimum(np.maximum(mv_0, 0), 255)).astype(np.uint8)
            mat.append(mv_0)
        mat = np.asarray(mat, dtype=np.float32)
        mv_origin = []
        return mat

if __name__ == '__main__':
    import time

    start = time.time()
    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            r'/home/sjhu/datasets/UCF-101-mpeg4',
            video_list= r'/home/sjhu/datasets/ucf101_split1_train.txt',
            num_segments=1,
            alpha=2,
            is_train=True,
        ),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    for i, (input_pairs, label) in enumerate(train_loader):
        iframe, mv= input_pairs
        # print(iframe.shape)
        # print(mv.shape)
        print("%d" % i)
    end = time.time()
    print("cost %f s" % ((end - start)))




