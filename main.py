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

from core.utils import Path as Pth
from experiments import classification

# for preparing data annotation
# from datasets import data_preparation
# data_preparation.reduce_annotation_classes()
# data_preparation.test_reduced_annotation()

# for training
classification = classification.Classification()
classification.train()

# for evaluation
# classification = exp_ss2.Classification()
# eval()
