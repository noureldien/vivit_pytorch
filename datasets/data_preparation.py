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


import os
import time
import cv2
import natsort
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from core import utils, image_utils, config_utils, pytorch_utils, consts
from datasets import data_sets
from datasets.transforms_3d import *
from core.utils import Path as Pth
from core.utils import TextLogger
from datasets.data_parser import Mp4Dataset

def reduce_annotation_classes():

    # how many classes to leave
    n_classes = 5

    root_path = Pth('videos/')
    json_labels_path = Pth("annotation/something-something-v2-labels.json")
    json_labels_mini_path = Pth("annotation/something-something-v2-labels-mini.json")

    json_data_tr_path = Pth("annotation/something-something-v2-train.json")
    json_data_vl_path = Pth("annotation/something-something-v2-validation.json")

    json_data_mini_tr_path = Pth("annotation/something-something-v2-mini-train.json")
    json_data_mini_vl_path = Pth("annotation/something-something-v2-mini-validation.json")

    json_data_mini_tr = __filter_classes(n_classes, json_data_tr_path, json_labels_path, root_path)
    json_data_mini_vl = __filter_classes(n_classes, json_data_vl_path, json_labels_path, root_path)

    utils.json_dump(json_data_mini_tr, json_data_mini_tr_path)
    utils.json_dump(json_data_mini_vl, json_data_mini_vl_path)

    json_labels_mini = {}
    idx = 0
    for item in json_data_mini_vl:
        item_template = item['template']
        item_template = __clean_template(item_template)
        if item_template not in json_labels_mini:
            json_labels_mini[item_template] = str(idx)
            idx += 1
    utils.json_dump(json_labels_mini, json_labels_mini_path)


def test_reduced_annotation():

    n_classes = 5
    root_path = Pth('videos/')
    json_labels_path = Pth("annotation/something-something-v2-labels.json")
    json_labels = utils.json_load(json_labels_path)

    json_data_mini_tr_path = Pth("annotation/something-something-v2-mini-train.json")
    json_data_mini_vl_path = Pth("annotation/something-something-v2-mini-validation.json")

    dataset_object_tr = Mp4Dataset(json_data_mini_tr_path, json_labels_path, root_path, is_test=False)
    classes_dict = dataset_object_tr.classes_dict

    json_data_mini_tr = utils.json_load(json_data_mini_tr_path)
    json_data_mini_vl = utils.json_load(json_data_mini_vl_path)

    classes_dict_tr = {}
    classes_dict_vl = {}

    for item in json_data_mini_tr:
        item_template = item['template']
        if item_template not in classes_dict_tr:
            classes_dict_tr[item_template] = 0
        classes_dict_tr[item_template] += 1


    for item in json_data_mini_vl:
        item_template = item['template']
        if item_template not in classes_dict_vl:
            classes_dict_vl[item_template] = 0
        classes_dict_vl[item_template] += 1

    print(len(classes_dict_tr))
    print(len(classes_dict_vl))

    print(np.sort(list(classes_dict_tr.keys())))
    print(classes_dict_tr.values())

    print(np.sort(list(classes_dict_vl.keys())))
    print(classes_dict_vl.values())

def __filter_classes(n_classes, json_data_src_path, json_labels_path, root_path):

    dataset_object = Mp4Dataset(json_data_src_path, json_labels_path, root_path, is_test=False)
    classes_dict = dataset_object.classes_dict

    json_src = utils.json_load(json_data_src_path)
    json_dst = []

    for item in json_src:
        item_template = item['template']
        item_template = __clean_template(item_template)
        item_class_idx = classes_dict[item_template]
        if item_class_idx < n_classes:
            json_dst.append(item)

    print(len(json_src))
    print(len(json_dst))

    return json_dst

def __clean_template(template):
    """ Replaces instances of `[something]` --> `something`"""
    template = template.replace("[", "")
    template = template.replace("]", "")
    return template