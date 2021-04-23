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
import json
import natsort
import time
import pytz
import h5py
import dill
import random
import numpy as np
import pickle as pkl
import pandas as pd
import cloudpickle as cpkl
import matplotlib.pyplot as plt
import scipy.io as sio
from datetime import datetime
from sklearn import preprocessing, manifold
from multiprocessing.dummy import Pool
import pathlib

from core import consts, plot_utils

# region Load and Dump

def pkl_load(path):
    with open(path, 'rb') as f:
        data = pkl.load(f, encoding='bytes')
    return data

def txt_load(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    lines = np.array(lines)
    return lines

def byte_load(path):
    with open(path, 'rb') as f:
        data = f.read()
    return data

def json_load(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data

def h5_load(path, dataset_name='data'):
    h5_file = h5py.File(path, 'r')
    data = h5_file[dataset_name].value
    h5_file.close()
    return data

def h5_load_multi(path, dataset_names):
    h5_file = h5py.File(path, 'r')
    data = [h5_file[name].value for name in dataset_names]
    h5_file.close()
    return data

def txt_dump(data, path):
    l = len(data) - 1
    with open(path, 'w') as f:
        for i, k in enumerate(data):
            if i < l:
                k = ('%s\n' % k)
            else:
                k = ('%s' % k)
            f.writelines(k)

def byte_dump(data, path):
    with open(path, 'wb') as f:
        f.write(data)

def pkl_dump(data, path, is_highest=True):
    with open(path, 'wb') as f:
        if not is_highest:
            pkl.dump(data, f)
        else:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

def json_dump(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def h5_dump(data, path, dataset_name='data'):
    h5_file = h5py.File(path, 'w')
    h5_file.create_dataset(dataset_name, data=data, dtype=data.dtype)
    h5_file.close()

def h5_dump_multi(data, dataset_names, path):
    h5_file = h5py.File(path, 'w')
    n_items = len(data)
    for i in range(n_items):
        item_data = data[i]
        item_name = dataset_names[i]
        h5_file.create_dataset(item_name, data=item_data, dtype=item_data.dtype)
    h5_file.close()

def csv_load(path, sep=',', header='infer'):
    df = pd.read_csv(path, sep=sep, header=header)
    data = df.values
    return data

def mat_load(path, m_dict=None):
    """
    Load mat files.
    :param path:
    :return:
    """
    if m_dict is None:
        data = sio.loadmat(path)
    else:
        data = sio.loadmat(path, m_dict)

    return data

def cloud_pkl_dump(data, path):
    with open(path, 'wb') as f:
        cpkl.dump(data, f)

def cloud_pkl_load(path):
    with open(path, 'rb') as f:
        data = cpkl.load(f)
    return data

def dill_dump(data, path):
    with open(path, 'wb') as f:
        dill.dump(data, f)

def dill_load(path):
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data

# endregion

# region File/Folder Names/Pathes

def file_names(path, is_natsort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names = os.walk(path).__next__()[2]

    if is_natsort:
        names = natsort.natsorted(names)

    return names

def file_names_filtered(path, extension, is_natsort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names_all = os.walk(path).__next__()[2]

    names = []
    extension = extension.lower()
    for n in names_all:
        extn = get_file_extension(n)
        if extn.lower() == extension:
            names.append(n)

    if is_natsort:
        names = natsort.natsorted(names)

    return names

def file_pathes(path, is_natsort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)
    names = os.walk(path).__next__()[2]

    if is_natsort:
        names = natsort.natsorted(names)

    pathes = ['%s/%s' % (path, n) for n in names]
    return pathes

def file_pathes_filtered(path, extension, is_natsort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)
    # breakpoint()
    names_all = os.walk(path).__next__()[2]

    names = []
    extension = extension.lower()
    for n in names_all:
        extn = get_file_extension(n)
        if extn.lower() == extension:
            names.append(n)

    if is_natsort:
        names = natsort.natsorted(names)

    pathes = ['%s/%s' % (path, n) for n in names]
    return pathes

def folder_names(path, is_natsort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names = os.walk(path).__next__()[1]

    if is_natsort:
        names = natsort.natsorted(names)

    return names

def folder_pathes(path, is_natsort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names = os.walk(path).__next__()[1]

    if is_natsort:
        names = natsort.natsorted(names)

    pathes = ['%s/%s' % (path, n) for n in names]
    return pathes

# endregion

# region Normalization

def normalize_mean_std(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x -= mean
    x /= std
    return x

def normalize_mean(x):
    mean = np.mean(x, axis=0)
    x /= mean
    return x

def normalize_sum(x):
    sum = np.sum(x, axis=1)
    x = np.array([x_i / sum_i for x_i, sum_i in zip(x, sum)])
    return x

def normalize_l2(x):
    return preprocessing.normalize(x)

def normalize_l1(x):
    return preprocessing.normalize(x, norm='l1')

def normalize_range_0_to_1(x):
    x = np.add(x, -x.min())
    x = np.divide(x, x.max())
    return x

# endregion

# region Array Helpers

def array_to_text(a, separator=', '):
    text = separator.join([str(s) for s in a])
    return text

def get_size_in_kb(size):
    size /= float(1024)
    return size

def get_size_in_mb(size):
    size /= float(1024 * 1024)
    return size

def get_size_in_gb(size):
    size /= float(1024 * 1024 * 1024)
    return size

def get_dtype_n_bytes_old(dtype):
    if dtype in [np.int8, np.uint8]:
        n_bytes = 1
    elif dtype in [np.int16, np.uint16, np.float16]:
        n_bytes = 2
    elif dtype in [np.int32, np.uint32, np.float32]:
        n_bytes = 4
    else:
        raise Exception('Sorry, unsupported dtype:', dtype)
    return n_bytes

def get_dtype_n_bytes(dtype):
    a = np.array((1,), dtype=dtype)
    n_bytes = a.nbytes
    return n_bytes

def get_array_memory_size(a):
    if type(a) is not np.ndarray:
        raise Exception('Sorry, input is not numpy array!')

    dtype = a.dtype
    n_bytes = get_dtype_n_bytes(dtype)

    s = a.size
    size = s * n_bytes
    return size

def get_expected_memory_size(array_shape, array_dtype):
    dtype = array_dtype
    n_bytes = get_dtype_n_bytes(dtype)
    s = 1
    for dim_size in array_shape:
        s *= dim_size

    size = s * n_bytes
    return size

def print_array(a):
    for item in a:
        print(item)

def print_array_joined(a):
    s = ', '.join([str(i) for i in a])
    print(s)

# endregion

# region Model Training/Testing

def debinarize_label(labels):
    debinarized = np.array([np.where(l == 1)[0][0] for l in labels])
    return debinarized

def binarize_label(labels, classes):
    binarized = preprocessing.label.label_binarize(labels, classes)
    return binarized

def get_model_feat_maps_info(model_type, feature_type):
    ex_feature = Exception('Sorry, unsupported feature type: %s' % (feature_type))
    ex_model = Exception('Sorry, unsupported model type: %s' % (model_type))
    info = None

    if model_type in ['vgg']:
        if feature_type in ['pool5']:
            info = 512, 7
        elif feature_type in ['conv5_3']:
            info = 512, 14
        else:
            raise ex_feature
    elif model_type in ['resnet18', 'resnet34', 'resnet18_cater']:
        if feature_type in ['conv5c']:
            info = 512, 7
        elif feature_type in ['conv5c_pool', 'convpool', 'conv_maxpool', 'conv_pool']:
            info = 512, 1
        else:
            raise ex_feature
    elif model_type in ['resnet3d', 'resnet152', 'resnet101', 'resnet50', 'resnet50_breakfast']:
        if feature_type in ['res4b35']:
            info = 1024, 14
        elif feature_type in ['res5c', 'res52', 'conv5c']:
            info = 2048, 7
        elif feature_type in ['pool5', 'conv5c_pool', 'conv_maxpool']:
            info = 2048, 1
        else:
            raise ex_feature
    elif model_type in ['i3d', 'i3d_breakfast', 'i3d_charades', 'i3d_cater']:
        if feature_type in ['mixed_5c']:
            info = 1024, 7
        elif feature_type in ['mixed_5c_maxpool']:
            info = 1024, 1
        elif feature_type in ['softmax']:
            info = 400, 1
        else:
            raise ex_feature
    elif model_type in ['i3d_kinetics_keras']:
        if feature_type in ['mixed_4f']:
            info = 832, 7
        else:
            raise ex_feature
    elif model_type in ['nl_50_charades', 'nl_101_charades']:
        if feature_type in ['conv_maxpool']:
            info = 2048, 1
        else:
            raise ex_feature
    elif model_type in ['mobilenetv3_small']:
        if feature_type in ['conv12']:
            info = 576, 7
        elif feature_type in ['convpool']:
            info = 576, 1
        else:
            raise ex_feature
    elif model_type in ['shufflenetv2_3d', 'shufflenetv2_3d_breakfast']:
        if feature_type in ['conv']:
            info = 1024, 7
        elif feature_type in ['conv_maxpool']:
            info = 1024, 1
        else:
            raise ex_feature
    else:
        raise ex_model

    return info

def get_model_n_frames_per_segment(model_type):
    if model_type in ['i3d']:
        n = 8
    elif model_type in ['shufflenetv2_3d']:
        n = 16
    elif model_type in ['resnet_2d']:
        n = 1
    else:
        raise Exception('Unknown Model Type %s ' % (model_type))

    return n

def calc_num_iterations(n_samples, batch_size):
    n_batch = int(n_samples / float(batch_size))
    n_batch = n_batch if n_samples % batch_size == 0 else n_batch + 1
    return n_batch

# endregion

# region Misc

def byte_array_to_string(value):
    decoded = string_decode(value)
    return decoded

def string_to_byte_array(value):
    decoded = string_encode(value)
    return decoded

def string_decode(value, coding_type='utf-8'):
    """
    Convert from byte array to string.
    :param value:
    :param coding_type:
    :return:
    """
    decoded = value.decode(coding_type)
    return decoded

def string_encode(value, coding_type='utf-8'):
    """
    Convert from byte string to array.
    :param value:
    :param coding_type:
    :return:
    """
    encoded = value.encode(coding_type)
    return encoded

def is_list(value):
    input_type = type(value)
    result = input_type is list
    return result

def is_tuple(value):
    input_type = type(value)
    result = input_type is tuple
    return result

def is_enumerable(value):
    input_type = type(value)
    result = (input_type is list or input_type is tuple)
    return result

def timestamp(local_timezone=False):
    """
    If local_timezone=True, then the timestamp becomes dependant on a the local timezone of the computer running the code.
    If local_timezone=False, then the timestep is according to the utc timezone.
    """

    time_now = datetime.now() if local_timezone else datetime.utcnow()
    time_stamp = '{0:%y}.{0:%m}.{0:%d}-{0:%H}:{0:%M}:{0:%S}'.format(time_now)

    return time_stamp

def remove_extension(name):
    if '.' not in name:
        raise Exception('No extension is found in this name %s' % (name))
    name = name.split('.')[:-1]
    name = ''.join(name)
    return name

def get_file_extension(name):
    name = name.split('.')[-1]
    return name

def print_counter(num, total, freq=None):
    if freq is None:
        print('... %d/%d' % (num, total))
    elif num % freq == 0:
        print('... %d/%d' % (num, total))

def make_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)

def empty_object_instance():
    """
    Initialize an instance of empty object.
    Helpful in debugging.
    """
    obj = type('', (), {})()
    return obj

def print_boxed_message(lines, width=100, is_bold=False):
    tl, tr, bl, br, h, v = ['╔', '╗', '╚', '╝', '═', '║'] if is_bold else ['┌', '┐', '└', '┘', '─', '│']

    width = np.max([len(l) for l in lines]) + 2 if width is None else width
    print(''.join([tl] + ([h] * width) + [tr]))
    for l in lines:
        n = len(l) + 2
        space = '' if width - n <= 0 else ''.join([' '] * (width - n))
        print(v + ' ' + l + space + ' ' + v)
    print(''.join([bl] + ([h] * width) + [br]))

def get_class_attributes(ClassName):
    attributes_all = dir(ClassName)
    attrs = []
    for att in attributes_all:
        if not (att.startswith('__') and att.endswith('__')):
            attrs.append(att)

    return attrs

# endregion

# region Classes

class Path(str):
    def __new__(self, relative_path, args=None, root_path=consts.DATA_ROOT_PATH):
        relative_path = relative_path % args if args is not None else relative_path
        path = os.path.join(root_path, relative_path)

        self.__path = path
        return self.__path

    def __str__(self):
        return self.__path

    def __repr__(self):
        return self.__path

class DurationTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def duration(self, is_string=True):
        stop_time = time.time()
        durtation = stop_time - self.start_time
        if is_string:
            durtation = self.format_duration(durtation)
        return durtation

    def format_duration(self, duration):
        if duration < 60:
            return str(duration) + " sec"
        elif duration < (60 * 60):
            return str(duration / 60) + " min"
        else:
            return str(duration / (60 * 60)) + " hr"

class TextLogger():
    def __init__(self, model_root_path=None, logging_type=consts.LOGGING_TYPES.both):

        self.logging_type = logging_type
        self.log_root_path = model_root_path
        self.log_file_path = None

        if model_root_path:
            if logging_type in [consts.LOGGING_TYPES.file, consts.LOGGING_TYPES.both]:
                self.log_file_path = Path('%s/logs_%s.txt', (model_root_path, timestamp()))
                make_directory(model_root_path)
        else:
            assert logging_type == consts.LOGGING_TYPES.console, 'Either provide model_root_path or choose to log only to console.'

    def print(self, text):

        if self.logging_type == consts.LOGGING_TYPES.console:
            self.print_to_console(text)
        elif self.logging_type == consts.LOGGING_TYPES.file:
            self.print_to_file(text)
        elif self.logging_type == consts.LOGGING_TYPES.both:
            self.print_to_console(text)
            self.print_to_file(text)
        else:
            raise Exception('Sorry, unknow logging type: %s' % (self.logging_type))

    def print_to_console(self, text):
        if self.logging_type in [consts.LOGGING_TYPES.console, consts.LOGGING_TYPES.both]:
            print(text)

    def print_to_file(self, text):
        if self.logging_type in [consts.LOGGING_TYPES.file, consts.LOGGING_TYPES.both]:
            text = str(text)
            log_file_path = self.log_file_path
            with open(log_file_path, 'a+') as f:
                f.write(text)
                f.write('\n')

# endregion

# region Main

def __init_console_logger():
    """
    Initalize the main console logger.
    """
    consts.CONSOLE_LOGGER = TextLogger(logging_type=consts.LOGGING_TYPES.console)

def main():
    __init_console_logger()

main()

# endregion
