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
import platform
import getpass
import numpy as np

# region Constants

DL_FRAMEWORK = None
PLATFORM = None
GPU_CORE_ID = 0
DATA_ROOT_PATH = None
PROJECT_ROOT_PATH = None
CONSOLE_LOGGER = None
MACHINE_NAME = platform.node()
USER_NAME = getpass.getuser()

IMAGENET_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CIFAR_RGB_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_RGB_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

# endregion

# region Structures: Users and Machines

class MACHINE_NAMES:
    desktop = 'u036713'
    server = 'nour-instance'

class ROOT_PATH_TYPES:
    dekstop_project = '/home/nour/Documents/PyCharmProjects/zipping_coding_challenge'
    desktop_data = '/home/nour/Documents/Datasets/Something-Something'
    server_project = '/home/agrawal/PyCharmProjects/zipping_coding_challenge'
    server_data = '/data/Something-Something'

# endregion

# region Structures: Data

class FRAME_SIZES:
    """
    Dimension of an image or frame given to the 2D CNN or 3D CNN.
    Only defined for two datasets: Kinetics, UCF101

    ucf101     , short size=128, crop to 112x112
    ucf101     , short size=146, crop to 128x128
    ucf101     , short size=256, crop to 224x224

    kinetics400, short size=130, crop to 112x112
    kinetics400, short size=150, crop to 128x128
    kinetics400, short size=262, crop to 224x224
    """

    ucf_small_crop = 112
    ucf_small_reSIZE = 120

    ucf_medium_crop = 128
    ucf_medium_resize = 146

    ucf_big_crop = 224
    ucf_big_resize = 256

    kinetics_small_crop = 112
    kinetics_small_resize = 130

    kinetics_medium_crop = 128
    kinetics_medium_resize = 150

    kinetics_big_crop = 224
    kinetics_big_resize = 262

class INPUT_IMAGE_DIMENSIONS:
    """
    Dimension of an image or frame given to the 2D CNN or 3D CNN.
    Only defined for two datasets: Kinetics, UCF101

    ucf101     , short size=128, crop to 112x112
    ucf101     , short size=146, crop to 128x128
    ucf101     , short size=256, crop to 224x224

    kinetics400, short size=130, crop to 112x112
    kinetics400, short size=150, crop to 128x128
    kinetics400, short size=262, crop to 224x224
    """

    ucf101_small = (112, 128)
    ucf101_medium = (128, 146)
    ucf101_big = (224, 256)

    kinetics_small = (112, 130)
    kinetics_medium = (128, 150)
    kinetics_big = (224, 262)

class RESIZE_TYPES:
    resize = 'resize'
    resize_crop = 'resize_crop'
    resize_crop_scaled = 'resize_crop_scaled'
    resize_keep_aspect_ratio_padded = 'resize_keep_aspect_ratio_padded'

# endregion

# region Structures: Visualization

class SEABORN_STYLES:
    dark = 'dark'
    white = 'white'
    ticks = 'ticks'
    whitegrid = 'whitegrid'
    darkgrid = 'darkgrid'

# endregion

# region Structures: Misc

class LOGGING_TYPES:
    console = 'console'
    file = 'file'
    both = 'both'

# endregion

# region Main

def __init_root_pathes():
    """
    Set the pathes of root directories.
    """
    global PROJECT_ROOT_PATH
    global DATA_ROOT_PATH

    if MACHINE_NAME == MACHINE_NAMES.desktop:
        PROJECT_ROOT_PATH = ROOT_PATH_TYPES.dekstop_project
        DATA_ROOT_PATH = ROOT_PATH_TYPES.desktop_data
    elif MACHINE_NAME == MACHINE_NAMES.server:
        PROJECT_ROOT_PATH = ROOT_PATH_TYPES.server_project
        DATA_ROOT_PATH = ROOT_PATH_TYPES.server_data
    else:
        raise NotImplementedError(f'Unknown username: {USER_NAME}')

def __init_console_logger():
    """
    Initalize the main console logger.
    """
    from core import utils
    global CONSOLE_LOGGER
    CONSOLE_LOGGER = utils.TextLogger(logging_type=LOGGING_TYPES.console)

def main():
    __init_root_pathes()

main()

# endregion
