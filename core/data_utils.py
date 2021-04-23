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

import numpy as np
import pickle as pkl
import h5py
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing, manifold

import os
import json
import natsort
import random
from multiprocessing.dummy import Pool
from core import utils, plot_utils, consts, config_utils

from core.utils import Path as Pth

# region Functions: Misc

def shuffle_uni(x):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]

    return x

def shuffle_bi(x, y):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y

def shuffle_tri(x, y, z):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    z = z[idx]

    return x, y, z

# endregion