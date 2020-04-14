"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""

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

VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'
## For Dataset
WIDTH = 256
HEIGHT = 340


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root,
                 video_list,
                 num_segments,
                 is_train):

        self._data_root = data_root
        self._num_segments = num_segments
        self._is_train = is_train
        self._iframe_scales = [1, .875, .75]
        self._mv_scales = [1, .875, .75, .66]
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
        self._input_mean = torch.from_numpy(
            np.array([0.43216, 0.394666, 0.37645]).reshape((3, 1, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.22803, 0.22145, 0.216989]).reshape((3, 1, 1, 1))).float()
        self._timescales = [1, 2, 4]
        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r') as f:
            for line in f:
                video, _, label = line.strip().split()
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                self._video_list.append((
                    video_path,
                    int(label)))

        print('%d videos loaded.' % len(self._video_list))

    def __getitem__(self, index):

        video, label = self._video_list[index]
        try:

            # shapes  (nums,height,width,channels)
            video_features = []
            extracter = VideoExtracter(video)

            # # process iframe
            iframes = extracter.load_keyframes(self._num_segments, self._is_train)
            iframes = self._iframe_transform(iframes) if self._is_train else self._infer_transform(iframes)
            iframes = np.asarray(iframes)
            iframes = np.transpose(iframes, (3, 0, 1, 2))
            iframes = torch.from_numpy(iframes).float() / 255.0
            iframes = (iframes - self._input_mean) / self._input_std

            # process mv and extract multi-scale
            mvs = extracter.load_mvs(self._num_segments, self._is_train)
            mv1 = random_sample(mvs, self._num_segments * self._timescales[0]) if self._is_train \
                else fix_sample(mvs, self._num_segments * self._timescales[0])

            mv2 = random_sample(mvs, self._num_segments * self._timescales[1]) \
                if self._is_train else fix_sample(mvs, self._num_segments * self._timescales[1])

            mv3 = random_sample(mvs, self._num_segments * self._timescales[2]) \
                if self._is_train else fix_sample(mvs, self._num_segments * self._timescales[2])

            mv1 = self.transform_normalize_mvs(mv1)
            mv2 = self.transform_normalize_mvs(mv2)
            mv3 = self.transform_normalize_mvs(mv3)

            # channels,depth, width, height
            assert iframes.shape[1] == self._num_segments, print("iframe shape wrong")
            assert mv1.shape[1] == self._num_segments * self._timescales[0], print("timesacle-1 shape wrong")
            assert mv2.shape[1] == self._num_segments * self._timescales[1], print("timescale-2 shape wrong")
            assert mv3.shape[1] == self._num_segments * self._timescales[2], print("timescale-3 shape wrong")
            return (iframes, mv1, mv2, mv3), label

        except Exception as e:
            traceback.print_exc()
            logging.exception(e)

    def __len__(self):
        return len(self._video_list)

    def transform_normalize_mvs(self, mvs):
        # preprocess mv
        mvs = np.asarray(mvs,dtype=np.float32)
        mvs = self._mv_transform(mvs) if self._is_train else self._infer_transform(mvs)
        mvs = np.asarray(mvs)
        mvs = np.transpose(mvs, (3, 0, 1, 2))
        mvs = torch.from_numpy(mvs).float() / 255.0
        mvs = (mvs - 0.5)
        # channels,depth, width, height
        return mvs


class VideoExtracter:
    def __init__(self, video_name):
        # ex: filename = 916710595466737253411014029368.mp4
        self.video_name = video_name
        # get basic decode information
        frames_type = coviexinfo.get_num_frames(video_name)
        self.num_frames = frames_type.shape[1]
        self.num_I = np.sum(frames_type[0] == 1).item()

    def load_keyframes(self, num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width, height, channels)
        """
        frames = coviexinfo.extract(self.video_name, 'get_I', self.num_frames, self.num_I, 0)
        if len(frames) == 0:
            mat = np.random.randint(255, size=(num_segments, WIDTH, HEIGHT, 3))
            return np.array(mat, dtype=np.float32)

        mat = []
        for i in range(self.num_I):
            rgb = np.dstack((frames[:, :, i * 3], frames[:, :, i * 3 + 1], frames[:, :, i * 3 + 2]))
            mat.append(rgb)
            # plt.imshow(rgb)
            # plt.show()
        mat = random_sample(mat, num_segments) if is_train else fix_sample(mat, num_segments)
        mat = np.asarray(mat, dtype=np.float32)
        return mat

    def load_mvs(self, num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width//4, height//4, channels=2) 0,255
        """
        # mv_ref_arr=(H/4,W/4,frames*6)
        # mv_ref_arr is a array with 3 dimensions. The first dimension denotes Height of a frame. The second dimension denotes Width of a frame.
        # For every frame, it contains mv_0_x, mv_0_y, ref_0, mv_1_x, mv_1_y, ref_1. So, the third dimension denote frames*6.

        mv_origin = coviexinfo.extract(self.video_name, 'get_mv', self.num_frames, self.num_I, 0)
        if len(mv_origin) == 0:
            mat = np.random.randint(1, size=(num_segments, WIDTH, HEIGHT, 2))
            return np.array(mat, dtype=np.float32)

        mat = []
        mv_0_x = mv_origin[:, :, ::6]
        mv_0_y = mv_origin[:, :, 1::6]
        for i in range(mv_0_x.shape[2]):
            mv_0 = np.dstack((mv_0_x[:, :, i], mv_0_y[:, :, i]))
            mv_0 = self.clip_and_scale(mv_0, 20)
            mat.append(mv_0 + 128)
            # plt.imshow(mv_0)
            # plt.show()

        return mat

    def clip_and_scale(self, img, size):
        return (img * (127.5 / size)).astype(np.int32)


if __name__ == '__main__':
    import time

    start = time.time()
    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            r'/home/sjhu/datasets/UCF-101-mpeg4',
            video_list=r'/home/sjhu/projects/pytorch-coviar/data/datalists/debug_train.txt',
            num_segments=10,
            is_train=True
        ),
        batch_size=1, shuffle=True,
        num_workers=8, pin_memory=False)

    for i, (input_pairs, label) in enumerate(train_loader):
        iframe, mv1, mv2, mv3 = input_pairs
        print("")
    end = time.time()
    print("cost %f s" % ((end - start)))
