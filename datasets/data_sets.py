# !/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

"""
The class VideoFolder is borrowed from:
https://github.com/TwentyBN/something-something-v2-baseline/blob/master/data_loader_skvideo.py
"""

import numpy as np
import subprocess
from skvideo.io import FFmpegReader

import torch
from torch.utils import data

from datasets.data_augmentor import Augmentor
from datasets.data_parser import Mp4Dataset
from core import utils, consts
from core.utils import Path as Pth

class VideoFolder(data.Dataset):

    def __init__(self, root, json_file_input, json_file_labels, clip_size,
                 n_clips, step_size, is_val, transform_pre=None, transform_post=None,
                 augmentation_mappings_json=None, augmentation_types_todo=None, is_test=False, framerate=None):

        self.dataset_object = Mp4Dataset(json_file_input, json_file_labels, root, is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.augmentor = Augmentor(augmentation_mappings_json, augmentation_types_todo)

        self.n_clips = n_clips
        self.clip_size = clip_size
        self.step_size = step_size
        self.is_val = is_val
        self.framerate = framerate

    def __getitem__(self, index):

        item = self.json_data[index]

        framerate_sampled = self.augmentor.jitter_fps(self.framerate)
        optional_args = {"-r": "%d" % framerate_sampled}

        duration = self.get_duration(item.path)

        if duration is not None:
            nframes = int(duration * framerate_sampled)
            optional_args["-vframes"] = "%d" % nframes

        # Open video file
        reader = FFmpegReader(item.path, inputdict={}, outputdict=optional_args)

        try:
            imgs = []
            for img in reader.nextFrame():
                imgs.append(img)
        except (RuntimeError, ZeroDivisionError) as exception:
            print('{}: MP4 reader cannot open {}. Empty list returned.'.format(type(exception).__name__, item.path))

        imgs = self.transform_pre(imgs)
        imgs, label = self.augmentor(imgs, item.label)
        imgs = self.transform_post(imgs)

        num_frames = len(imgs)
        target_idx = self.classes_dict[label]

        if self.n_clips > -1:
            num_frames_necessary = self.clip_size * self.n_clips * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 0
        if num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)

        imgs = imgs[offset: num_frames_necessary + offset: self.step_size]

        if len(imgs) < (self.clip_size * self.n_clips):
            imgs.extend([imgs[-1]] * ((self.clip_size * self.n_clips) - len(imgs)))

        # stack images
        data = torch.stack(imgs) # (T, C, H, W)
        data = data.permute(1, 0, 2, 3) # (C, T, H, W)

        return (data, target_idx)

    def __len__(self):
        n_items = len(self.json_data)
        return n_items

    def get_duration(self, file):

        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        duration = float(result.stdout)
        return duration
