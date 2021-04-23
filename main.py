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

Main file of the project.
"""

import os
import sys
import traceback
import warnings
import numpy as np

from core import utils, config_utils, image_utils, consts
from core.utils import Path as Pth
from experiments import exp_classification
from scripts.train import train
from scripts.eval import eval

utils.print_boxed_message([
    'THIS IMPLEMENTATION IS USING PYTORCH',
    'HOWEVER, THE CODE ASLO PROVIDES KERAS IMPLEMENTATION'])


# for training
train()

# for evaluation
# eval()
