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
Documentation
"""

import os
import time
import cv2
import natsort
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets

from core import utils, image_utils, config_utils, pytorch_utils, consts
from datasets import data_sets, transforms_3d
from core.utils import Path as Pth
from core.utils import TextLogger

# region Initializations

np.random.seed(0)
random.seed(0)

# endregion

# region Data Loaders

class DataLoader3D():

    def __init__(self, n_frames):
        super(DataLoader3D, self).__init__()

        self.batch_size_tr = 32
        self.batch_size_te = 32
        self.n_workers_tr = 8
        self.n_workers_te = 8
        self.img_dim_resize = 256
        self.img_dim_crop = 224
        self.upscale_factor_train = 1.4
        self.upscale_factor_val = 1.0
        self.n_clips = 1
        self.clip_size = n_frames
        self.step_size = 1.0
        self.framerate = 12

        self.data_folder = Pth('videos/')
        self.json_data_train = Pth("annotation/something-something-v2-train.json")
        self.json_data_val = Pth("annotation/something-something-v2-validation.json")
        self.json_data_test = Pth("annotation/something-something-v2-test.json")
        self.json_file_labels = Pth("annotation/something-something-v2-labels.json")
        self.augmentation_mappings = Pth("annotation/augmentation-mappings.json")
        self.augmentation_types_todo = ["left/right", "left/right agnostic", "jitter_fps"]
        self.transform_pre_train = None
        self.transform_pre_val = None
        self.transform_post = None

        self.norm_mean = consts.IMAGENET_RGB_MEAN
        self.norm_std = consts.IMAGENET_RGB_STD

        ###########
        # "num_classes": 174,
        # "clip_size": 72,

    def initialize(self):
        # we use custom dataset
        dataset_tr, dataset_te = self.__get_datasets()

        n_tr = len(dataset_tr)
        n_te = len(dataset_te)

        # data loaders
        loader_tr = DataLoader(dataset_tr, batch_size=self.batch_size_tr, num_workers=self.n_workers_tr, pin_memory=True, drop_last=True, shuffle=True)
        loader_te = DataLoader(dataset_te, batch_size=self.batch_size_te, num_workers=self.n_workers_te, pin_memory=True, drop_last=True, shuffle=False)

        return (loader_tr, loader_te, n_tr, n_te)

    def __get_datasets(self, ):
        n_clips = self.n_clips
        clip_size = self.clip_size
        step_size = self.step_size
        framerate = self.framerate

        dataset_tr = data_sets.VideoFolder(root=self.data_folder,
                                           json_file_input=self.json_data_train,
                                           json_file_labels=self.json_file_labels,
                                           clip_size=clip_size,
                                           nclips=n_clips,
                                           step_size=step_size,
                                           framerate=framerate,
                                           is_val=False,
                                           transform_pre=self.transform_pre_train,
                                           transform_post=self.transform_post,
                                           augmentation_mappings_json=self.augmentation_mappings,
                                           augmentation_types_todo=self.augmentation_types_todo,
                                           get_item_id=False)

        dataset_te = data_sets.VideoFolder(root=self.data_folder,
                                         json_file_input=self.json_data_val,
                                         json_file_labels=self.json_file_labels,
                                         clip_size=clip_size,
                                         nclips=n_clips,
                                         step_size=step_size,
                                         framerate=framerate,
                                         is_val=True,
                                         transform_pre=self.transform_pre_val,
                                         transform_post=self.transform_post,
                                         get_item_id=True)

        return (dataset_tr, dataset_te)

    def __get_transforms(self):
        transforms_tr = transforms.Compose([transforms_3d.RandomResizedCrop(self.img_dim_resize, self.img_dim_crop, p=1.0, consistent=True),
                                            transforms_3d.RandomHorizontalFlip(consistent=True),
                                            transforms_3d.ToTensor(),
                                            transforms_3d.Normalize(self.norm_mean, self.norm_std),
                                            transforms_3d.Stack(dim=1)])

        transforms_te = transforms.Compose([transforms_3d.Resize(size=self.img_dim_resize),
                                            transforms_3d.CenterCrop(size=self.img_dim_crop),
                                            transforms_3d.ToTensor(),
                                            transforms_3d.Normalize(self.norm_mean, self.norm_std),
                                            transforms_3d.Stack(dim=1)])

        return (transforms_tr, transforms_te)

# endregion
