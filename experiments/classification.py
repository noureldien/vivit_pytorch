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
Experiments on contrastive learning form videos.
"""

import datetime
import yaml
import cv2
import numpy as np

from core import utils, plot_utils, consts, data_utils, pytorch_utils, pytorch_learners
from core.utils import TextLogger, Path as Pth
from datasets import data_loaders
from modules.vivit import ViViT
import modules

class Classification():

    def __init__(self):
        super().__init__()
        pass

    def train(self, ):
        """
        Train model on downstream task.
        config: the configuration object with all the parameters to train the model.
        best_pretext_epoch: optional arg for init the model.
        """

        model_name = 'ss2_vivit_%s' % (utils.timestamp())
        model_root_path = Pth('models/%s', (model_name,))
        n_epochs = 100
        clip_size = 16
        batch_size = 32
        n_workers = 8
        n_classes = 5
        # n_classes = 174
        input_shape = (3, clip_size, 224, 224)

        # building data
        loader_tr, loader_te, n_tr, n_te = data_loaders.DataLoader3D(n_classes, batch_size, clip_size, n_workers).initialize()

        # building the model
        model = ViViT(num_classes=n_classes, clip_size=clip_size)
        model = model.cuda()
        pytorch_utils.model_summary(model, input_size=input_shape, batch_size=-1, device='cuda')

        # callbacks
        callbacks = []
        callbacks.append(pytorch_utils.ModelSaveCallback(model, model_root_path))

        # train
        learner = pytorch_learners.ClassifierLearner(model, model._optimizer, model._loss_fn, model._metric_fn, callbacks)
        learner.train(loader_tr, loader_te, n_tr, n_te, n_epochs, batch_size, batch_size)

        print('--- finish time')
        print(datetime.datetime.now())

    def eval(self, ):
        """
        Test model on downstream task.
        """

        raise NotImplementedError()

        batch_size_te = config.TEST.BATCH_SIZE
        epoch_num = config.TRAIN.N_EPOCHS
        model_name = config.MODEL.NAME

        dataset_name = consts.DATASET_TYPES.ucf101
        img_dim_crop = config.INPUT.IMAGE_DIM_CROP
        base_model_path = ''
        n_segments = 2

        n_fps = config.INPUT.N_FPS
        input_shape = (n_segments, 3, n_fps, img_dim_crop, img_dim_crop)
        model_path = Pth('%s/models/%s/%03d.pt', (dataset_name, model_name, epoch_num))
        logger = utils.TextLogger(logging_type=consts.LOGGING_TYPES.console)

        logger.print('--- start time')
        logger.print(datetime.datetime.now())
        logger.print(model_name)

        # building data
        _, loader_te, n_tr, n_te = data_loaders.DataLoader3D(config, logger).initialize()

        # building the model
        model = DownstreamModel(config, logger, base_model_path, n_tr)
        pytorch_utils.load_model_dict(model, model_path, strict=True, resolve_multi_gpu=True)
        pytorch_utils.freeze_model(model)
        pytorch_utils.model_summary(model, input_size=input_shape, batch_size=-1, device='cpu', logger=logger)
        model = pytorch_utils.parallelize_model(model, config.GPU_IDS)
        model = model.cuda()

        # test
        learner = pytorch_learners.ClassifierLearner(config, model, model._optimizer, model._scheduler, model._loss_fn, model._metric_fn, logger)
        learner.test(loader_te, n_te, batch_size_te)

        logger.print('--- finish time')
        logger.print(datetime.datetime.now())

