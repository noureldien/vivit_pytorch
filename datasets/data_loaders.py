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


import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import data_sets
from datasets.transforms_3d import *
from core.utils import Path as Pth

# region Initializations

np.random.seed(0)
random.seed(0)

# endregion

# region Data Loaders

class DataLoader3D():

    def __init__(self, n_classes, batch_size, clip_size, n_workers):
        super(DataLoader3D, self).__init__()

        self.batch_size = batch_size
        self.n_workers = n_workers
        self.img_dim_resize = 256
        self.img_dim_crop = 224
        self.upscale_factor_train = 1.4
        self.upscale_factor_val = 1.0
        self.clip_size = clip_size
        self.n_clips = 1
        self.step_size = 1
        self.framerate = 12

        self.data_folder = Pth('videos/')
        self.augmentation_mappings = Pth("annotation/augmentation-mappings.json")
        self.augmentation_types_todo = ["left/right", "left/right agnostic", "jitter_fps"]
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        assert n_classes in [174, 5]
        if n_classes == 174:
            self.json_labels = Pth("annotation/something-something-v2-labels.json")
            self.json_data_tr = Pth("annotation/something-something-v2-train.json")
            self.json_data_vl = Pth("annotation/something-something-v2-validation.json")
        elif n_classes == 5:
            self.json_labels = Pth("annotation/something-something-v2-labels-mini.json")
            self.json_data_tr = Pth("annotation/something-something-v2-mini-train.json")
            self.json_data_vl = Pth("annotation/something-something-v2-mini-validation.json")
        else:
            raise Exception('Unknown number of classes: %d' % (n_classes))


    def initialize(self):
        # we use custom dataset
        dataset_tr, dataset_vl = self.__get_datasets()

        # data loaders
        loader_tr = DataLoader(dataset_tr, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True, drop_last=True, shuffle=True)
        loader_vl = DataLoader(dataset_vl, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True, drop_last=True, shuffle=False)

        n_tr = len(dataset_tr)
        n_vl = len(dataset_vl)

        return (loader_tr, loader_vl, n_tr, n_vl)

    def __get_datasets(self, ):

        # define augmentation pipeline
        self.transform_pre_tr = ComposeMix([[RandomRotationVideo(15), "vid"],[Scale(self.img_dim_resize), "img"],[RandomCropVideo(self.img_dim_crop), "vid"],])
        self.transform_pre_vl = ComposeMix([[Scale(self.img_dim_resize), "img"],[transforms.ToPILImage(), "img"],[transforms.CenterCrop(self.img_dim_crop), "img"],])
        self.transform_post = ComposeMix([[transforms.ToTensor(), "img"],[transforms.Normalize(mean=self.norm_mean,  std=self.norm_std), "img"]])

        dataset_tr = data_sets.VideoFolder(root=self.data_folder,
                                           json_file_input=self.json_data_tr,
                                           json_file_labels=self.json_labels,
                                           clip_size=self.clip_size,
                                           n_clips=self.n_clips,
                                           step_size=self.step_size,
                                           framerate=self.framerate,
                                           is_val=False,
                                           transform_pre=self.transform_pre_tr,
                                           transform_post=self.transform_post,
                                           augmentation_mappings_json=self.augmentation_mappings,
                                           augmentation_types_todo=self.augmentation_types_todo)

        dataset_vl = data_sets.VideoFolder(root=self.data_folder,
                                         json_file_input=self.json_data_vl,
                                         json_file_labels=self.json_labels,
                                         clip_size=self.clip_size,
                                         n_clips=self.n_clips,
                                         step_size=self.step_size,
                                         framerate=self.framerate,
                                         is_val=True,
                                         transform_pre=self.transform_pre_vl,
                                         transform_post=self.transform_post)

        return (dataset_tr, dataset_vl)

# endregion
