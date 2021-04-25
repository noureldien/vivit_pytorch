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
import sys
import time

import torch
import torch.onnx

from modules import layers
from core import utils, data_utils, config_utils, consts, metrics, pytorch_utils
from core.utils import Path as Pth
from core.utils import TextLogger
import pathlib

class ClassifierLearner():

    def __init__(self, model, optimizer=None, loss_fn=None, metric_fn=None, callbacks=()):
        super(ClassifierLearner).__init__()

        self.model = model
        self.optimizer = optimizer

        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.callbacks = callbacks
        self.logger = TextLogger(logging_type=consts.LOGGING_TYPES.console)

    def train(self, loader_tr, loader_te, n_tr, n_te, n_epochs, batch_size_tr, batch_size_te):
        """
        Train using input sampler. Each epoch, we sample inputs from the sampler
        """

        model = self.model
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        metric_fn = self.metric_fn
        logger = self.logger
        callbacks = self.callbacks
        n_gpus = 1

        n_iter_tr = utils.calc_num_iterations(n_tr, batch_size_tr)
        n_iter_te = utils.calc_num_iterations(n_te, batch_size_te)

        logger.print('... [tr]: n, n_iter, batch_size: %d, %d, %d' % (n_tr, n_iter_tr, batch_size_tr))
        logger.print('... [te]: n, n_iter, batch_size: %d, %d, %d' % (n_te, n_iter_te, batch_size_te))
        logger.print('')

        acc_max_tr = 0.0
        acc_max_te = 0.0

        # loop on epochs
        for idx_epoch in range(n_epochs):

            epoch_num = idx_epoch + 1
            loss_tr = 0.0
            loss_te = 0.0
            acc_tr = 0.0
            acc_te = 0.0
            tt1 = time.time()

            # switch to training mode
            model.train()

            # loop on batches for train
            for idx_batch, (t_x_tr_b, t_y_tr_b) in enumerate(loader_tr):
                batch_num = idx_batch + 1

                # copy to gpu
                t_x_tr_b = t_x_tr_b.cuda()
                t_y_tr_b = t_y_tr_b.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # feedforward + loss
                t_y_tr_pred_b = model(t_x_tr_b)
                t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

                # backward + optimize
                t_loss_b.backward()
                optimizer.step()
                loss_b = t_loss_b.item()

                # loss + accuracy
                acc_b = metric_fn(t_y_tr_pred_b, t_y_tr_b)
                loss_tr += loss_b
                acc_tr += acc_b
                loss_b = loss_tr / float(batch_num)
                acc_b = 100 * acc_tr / float(batch_num)

                # calling the callbacks
                if callbacks is not None:
                    for cb in callbacks:
                        cb.on_batch_ends(batch_num, is_training=True)

                tt2 = time.time()
                duration = tt2 - tt1
                sys.stdout.write('\r%04ds | epoch %02d/%02d | b_tr %02d/%02d | l_tr %.02f | m_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_iter_tr, loss_b, acc_b))

            # switch to eval mode
            with torch.no_grad():
                model.eval()

                # loop on batches for test
                for idx_batch, (t_x_te_b, t_y_te_b) in enumerate(loader_te):
                    batch_num = idx_batch + 1

                    # copy to gpu
                    t_x_te_b = t_x_te_b.cuda()
                    t_y_te_b = t_y_te_b.cuda()

                    #  nograd + feedforward + loss
                    t_y_te_pred_b = model(t_x_te_b)
                    t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)

                    # loss + accuracy
                    loss_b = t_loss_b.item()
                    acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
                    loss_te += loss_b
                    acc_te += acc_b
                    loss_b = loss_te / float(batch_num)
                    acc_b = 100 * acc_te / float(batch_num)

                    # calling the callbacks
                    if callbacks is not None:
                        for cb in callbacks:
                            cb.on_batch_ends(batch_num, is_training=False)

                    tt2 = time.time()
                    duration = tt2 - tt1
                    sys.stdout.write('\r%04ds | epoch %02d/%02d | b_vl %02d/%02d | l_vl %.02f | m_vl %02.02f    ' % (duration, epoch_num, n_epochs, batch_num, n_iter_te, loss_b, acc_b))

            tt2 = time.time()
            duration = tt2 - tt1

            loss_tr /= float(n_iter_tr)
            loss_te /= float(n_iter_te)
            acc_tr = 100 * acc_tr / float(n_iter_tr)
            acc_te = 100 * acc_te / float(n_iter_te)
            acc_max_tr = max(acc_max_tr, acc_tr)
            acc_max_te = max(acc_max_te, acc_te)
            learning_rate = pytorch_utils.get_learning_rates(optimizer, n_gpus)[0]

            epoch_log = '%04ds | epoch %02d/%02d | l_tr %.02f | l_vl %.02f | m_tr %02.02f | m_vl %02.02f | mm_tr %02.02f | mm_te %02.02f | lr %.04f' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te, learning_rate)
            sys.stdout.write('\r' + epoch_log + '\n')
            logger.print_to_file(epoch_log)

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_ends(epoch_num)

    def test(self, loader_te, n_te, batch_size_te):
        """
        Test using input sampler. Each epoch, we sample inputs from the sampler
        """

        model = self.model
        loss_fn = self.loss_fn
        metric_fn = self.metric_fn
        callbacks = self.callbacks
        logger = self.logger

        n_iter_te = utils.calc_num_iterations(n_te, batch_size_te)
        logger.print('... [te]: n, n_iter, batch_size: %d, %d, %d' % (n_te, n_iter_te, batch_size_te))
        logger.print('')

        acc_max_te = 0.0
        loss_te = 0.0
        acc_te = 0.0
        tt1 = time.time()

        # switch to eval mode
        model.eval()
        model.training = False

        with torch.no_grad():

            # loop on batches for test
            for idx_batch, (t_x_te_b, t_y_te_b) in enumerate(loader_te):
                batch_num = idx_batch + 1

                # copy to gpu
                t_x_te_b = t_x_te_b.cuda()
                t_y_te_b = t_y_te_b.cuda()

                #  nograd + feedforward + loss
                t_y_te_pred_b = model(t_x_te_b)
                t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)

                # loss + accuracy
                loss_b = t_loss_b.item()
                acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
                loss_te += loss_b
                acc_te += acc_b
                loss_b = loss_te / float(batch_num)
                acc_b = 100 * loss_te / float(batch_num)

                # calling the callbacks
                if callbacks is not None:
                    for cb in callbacks:
                        cb.on_batch_ends(batch_num, is_training=False)

                tt2 = time.time()
                duration = tt2 - tt1
                sys.stdout.write('\r%04ds | b_vl %02d/%02d | l_vl %.02f | m_vl %02.02f    ' % (duration, batch_num, n_iter_te, loss_b, acc_b))

            tt2 = time.time()
            duration = tt2 - tt1

            loss_te /= float(n_iter_te)
            acc_te = 100 * acc_te / float(n_iter_te)
            acc_max_te = max(acc_max_te, acc_te)

            epoch_log = '%04ds | l_vl %.02f | m_vl %02.02f | mm_te %02.02f' % (duration, loss_te, acc_te, acc_max_te)
            sys.stdout.write('\r' + epoch_log + '\n')
            logger.print_to_file(epoch_log)

    def test_both(self, loader_tr, loader_te, n_tr, n_te, batch_size_tr, batch_size_te):
        """
        Train using input sampler. Each epoch, we sample inputs from the sampler
        """

        model = self.model
        loss_fn = self.loss_fn
        metric_fn = self.metric_fn
        logger = self.logger
        callbacks = self.callbacks

        n_iter_tr = utils.calc_num_iterations(n_tr, batch_size_tr)
        n_iter_te = utils.calc_num_iterations(n_te, batch_size_te)

        logger.print('... [tr]: n, n_iter, batch_size: %d, %d, %d' % (n_tr, n_iter_tr, batch_size_tr))
        logger.print('... [te]: n, n_iter, batch_size: %d, %d, %d' % (n_te, n_iter_te, batch_size_te))
        logger.print('')

        acc_max_tr = 0.0
        acc_max_te = 0.0

        epoch_num = 1
        n_epochs = 1

        # switch to eval mode
        model.eval()
        model.training = False

        with torch.no_grad():

            loss_tr = 0.0
            loss_te = 0.0
            acc_tr = 0.0
            acc_te = 0.0
            tt1 = time.time()

            # loop on batches for train
            for idx_batch, (t_x_tr_b, t_y_tr_b) in enumerate(loader_tr):
                batch_num = idx_batch + 1

                # copy to gpu
                t_x_tr_b = t_x_tr_b.cuda()
                t_y_tr_b = t_y_tr_b.cuda()

                #  nograd + feedforward + loss
                t_y_tr_pred_b = model(t_x_tr_b)
                t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

                # loss + accuracy
                loss_b = t_loss_b.item()
                acc_b = metric_fn(t_y_tr_pred_b, t_y_tr_b)
                loss_tr += loss_b
                acc_tr += acc_b
                loss_b = loss_tr / float(batch_num)
                acc_b = 100 * acc_tr / float(batch_num)

                # calling the callbacks
                if callbacks is not None:
                    for cb in callbacks:
                        cb.on_batch_ends(batch_num, is_training=True)

                tt2 = time.time()
                duration = tt2 - tt1
                sys.stdout.write('\r%04ds | epoch %02d/%02d | b_tr %02d/%02d | l_tr %.02f | m_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_iter_tr, loss_b, acc_b))

            # loop on batches for test
            for idx_batch, (t_x_te_b, t_y_te_b) in enumerate(loader_te):
                batch_num = idx_batch + 1

                # copy to gpu
                t_x_te_b = t_x_te_b.cuda()
                t_y_te_b = t_y_te_b.cuda()

                #  nograd + feedforward + loss
                t_y_te_pred_b = model(t_x_te_b)
                t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)

                # loss + accuracy
                loss_b = t_loss_b.item()
                acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
                loss_te += loss_b
                acc_te += acc_b
                loss_b = loss_te / float(batch_num)
                acc_b = 100 * acc_te / float(batch_num)

                # calling the callbacks
                if callbacks is not None:
                    for cb in callbacks:
                        cb.on_batch_ends(batch_num, is_training=False)

                tt2 = time.time()
                duration = tt2 - tt1
                sys.stdout.write('\r%04ds | epoch %02d/%02d | b_vl %02d/%02d | l_vl %.02f | m_vl %02.02f    ' % (duration, epoch_num, n_epochs, batch_num, n_iter_te, loss_b, acc_b))

            tt2 = time.time()
            duration = tt2 - tt1

            loss_tr /= float(n_iter_tr)
            loss_te /= float(n_iter_te)
            acc_tr = 100 * acc_tr / float(n_iter_tr)
            acc_te = 100 * acc_te / float(n_iter_te)
            acc_max_tr = max(acc_max_tr, acc_tr)
            acc_max_te = max(acc_max_te, acc_te)
            learning_rate = 0.0

            epoch_log = '%04ds | epoch %02d/%02d | l_tr %.02f | l_vl %.02f | m_tr %02.02f | m_vl %02.02f | mm_tr %02.02f | mm_te %02.02f | lr %.04f' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te, learning_rate)
            sys.stdout.write('\r' + epoch_log + '\n')
            logger.print_to_file(epoch_log)
