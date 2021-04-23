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
Configurations for project.
"""

import os
import sys
import platform
import argparse
import yaml
import pprint
import ast

from core import utils, consts
from core.utils import Path as Pth

# region Functions

def get_machine_name():
    return platform.node()

def __config_python_version():
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")
    else:
        # print('correct python version')
        pass

# run configs
__config_python_version()

# endregion
