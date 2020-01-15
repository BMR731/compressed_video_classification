"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
import cv2
from coviar import get_num_frames
from coviar import load
from transforms import *
import torchvision
from transforms_hsj import *
from PIL import Image
import matplotlib.pyplot as plt
GOP_SIZE = 12


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
    def __init__(self, data_root,
                 video_list,
                 representation,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._num_segments = num_segments
        self._representation = representation #iframe 0; mv 1 ,residual 2
        self._is_train = is_train
        self._accumulate = True

        # modify depend on Kinetics-400 dataset setting
        self._input_mean = torch.from_numpy(
            np.array([0.43216, 0.394666, 0.37645]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.22803, 0.22145, 0.216989]).reshape((1, 3, 1, 1))).float()

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

    def _get_train_frame_index(self, num_frames, seg ,representation):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                           representation=representation)

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

        return get_gop_pos(v_frame_idx, self._representation)

    def __getitem__(self, index):
        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        iframes = []
        mvs = []
        for seg in range(self._num_segments):

            ######### for img
            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg,'iframe')
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg,'iframe')

            img = load(video_path, gop_index, gop_pos, 0, self._accumulate)
            if img is None:
                print('Error: loading video %s failed.' % video_path)
                img = np.zeros((256, 256, 3))
            img = color_aug(img)
            # BGR to RGB. (PyTorch uses RGB according to doc.)
            img = img[..., ::-1]

            ######## for mv
            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg,'mv')
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg,'mv')
            mv = load(video_path, gop_index, gop_pos, 1, self._accumulate)
            if mv is None:
                print('Error: loading video %s failed.' % video_path)
                mv = np.zeros((256, 256, 2))

            mv = clip_and_scale(mv, 20)
            mv += 128
            mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)

            iframes.append(img)
            mvs.append(mv)

        iframes = self.normalization(iframes, 'iframe')
        mvs = self.normalization(mvs, 'mv')
        #      (iframes,mvs),label (depth, channel,width,height)
        return (iframes, mvs), label

    def __len__(self):
        return len(self._video_list)

    def normalization(self, video_features, representation):
        video_features = np.array(video_features, dtype=np.uint8)
        output = []
        if representation == 'iframe':
            for i in range(video_features.shape[0]):
                t = video_features[i, ...]
                img = Image.fromarray(t)
                output.append(transform_rgb_residual(img) if self._is_train else transform_infer(img))
            output = torch.stack(output)
            output = np.transpose(output, (0, 1, 2, 3)) / 255.0
            output = (output - self._input_mean) / self._input_std
        elif representation == 'mv':
            # torch.Size([3, 2, 224, 224])
            for i in range(video_features.shape[0]):
                t = video_features[i, ...]
                padding = np.zeros(shape=(t.shape[0], t.shape[1], 1), dtype=np.uint8)
                t = np.dstack((t,padding))
                img = Image.fromarray(t)
                visualize_mv(t)
                output.append(transform_mv(img) if self._is_train else transform_infer(img))
            output = torch.stack(output)
            output = np.transpose(output, (0, 1, 2, 3)) / 255.0
            output = (output - output.mean())[:, :2, :, :]
        else:
            assert False, print("representation wrong")

        return output

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