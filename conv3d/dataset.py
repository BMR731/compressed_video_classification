"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.models.resnet
from coviar import get_num_frames
from coviar import load
from transforms import *
import torchvision

GOP_SIZE = 12

IFRAME = 0
MV = 1
RESIDUAL = 2


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(n, num_segments, seg, representation):
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg + 1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        # Exclude the 0-th frame, because it's an I-frame.
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
    def __init__(self, data_root,
                 video_list,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._num_segments = num_segments
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
        self._is_train = is_train
        self._accumulate = accumulate
        ## use kinetic pretrain setting
        self._input_mean = torch.from_numpy(
            np.array([0.43216, 0.394666, 0.37645]).reshape((3, 1, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.22803, 0.22145, 0.216989]).reshape((3, 1, 1, 1))).float()

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r') as f:
            for line in f:
                video, _, label = line.strip().split()
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                self._video_list.append((
                    video_path,
                    int(label),
                    get_num_frames(video_path)))

        print('%d videos loaded.' % len(self._video_list))

    def _get_train_frame_index(self, num_frames, seg, representation):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg, representation)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, representation)

    def _get_test_frame_index(self, num_frames, seg, representation):
        if representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
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
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg, 'iframe')
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg, 'iframe')

            img = load(video_path, gop_index, gop_pos, IFRAME, self._accumulate)
            if img is None:
                print('Error: loading video %s failed.' % video_path)
                img = np.zeros((256, 256, 3))
            # img = color_aug(img)
            # BGR to RGB. (PyTorch uses RGB according to doc.)
            img = img[..., ::-1]

            # for mv .notice here we should use the same gop_index with iframe
            mv = load(video_path, gop_index, gop_pos+3, MV, self._accumulate)
            if mv is None:
                print('Error: loading video %s failed.' % video_path)
                mv = np.zeros((256, 256, 2))

            mv = clip_and_scale(mv, 20)  # scale up the value
            mv += 128
            mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)

            iframes.append(img)
            mvs.append(mv)

        # preprocess iframe
        iframes = self._iframe_transform(iframes) if self._is_train else self._infer_transform(iframes)
        iframes = np.array(iframes)
        iframes = np.transpose(iframes, (3, 0, 1, 2))
        iframes = torch.from_numpy(iframes).float() / 255.0
        iframes = (iframes - self._input_mean) / self._input_std

        # preprocess mv
        mvs = self._mv_transform(mvs) if self._is_train else self._infer_transform(mvs)
        mvs = np.array(mvs)
        mvs = np.transpose(mvs, (3, 0, 1, 2))
        mvs = torch.from_numpy(mvs).float() / 255.0
        mvs = (mvs - 0.5)
        # channels,depth, width, height
        return (iframes, mvs), label

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
