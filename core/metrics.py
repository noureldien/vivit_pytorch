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
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

def top_n_score(n_top, y, y_cap):
    n_corrects = 0
    for gt, pr in zip(y, y_cap):
        idx = np.argsort(pr)[::-1]
        idx = idx[0:n_top]
        gt = np.where(gt == 1)[0][0]
        if gt in idx:
            n_corrects += 1
    n = len(y)
    score = n_corrects / float(n)
    return score

def accuracy(y_pred, y_true):
    n_y = y_true.size(0)
    idx_predicted = torch.argmax(y_pred, 1)
    acc = (idx_predicted == y_true).sum().item()
    acc = acc / float(n_y)
    return acc

def binary_accuracy(y_pred, y_true):
    n_y = y_true.size(0)
    y_pred = torch.squeeze(y_pred)
    y_true = torch.squeeze(y_true)
    y_pred_thresholded = torch.zeros_like(y_pred)
    y_pred_thresholded[y_pred >= 0.5] = 1

    acc = (y_pred_thresholded == y_true)
    acc = torch.sum(acc).item()
    acc = acc / float(n_y)
    return acc

def binary_siamese(y_pred, y_true):
    n_y = y_true.size(0)
    y_pred = torch.squeeze(y_pred)
    y_true = torch.squeeze(y_true)
    y_pred_thresholded = torch.zeros_like(y_pred)
    y_pred_thresholded[y_pred <= 0.5] = 1.0

    acc = (y_pred_thresholded == y_true)
    acc = torch.sum(acc).item()
    acc = acc / float(n_y)
    return acc

def binary_hinge(y_pred, y_true):
    n_y = y_true.size(0)
    y_pred = torch.squeeze(y_pred)
    y_true = torch.squeeze(y_true)
    y_pred_thresholded = torch.zeros_like(y_pred)
    y_pred_thresholded[y_pred <= 0.0] = 1
    y_pred_thresholded[y_pred > 0.0] = -1

    acc = (y_pred_thresholded == y_true)
    acc = torch.sum(acc).item()
    acc = acc / float(n_y)
    return acc

def cosine_similarity_accuracy(y_pred, y_true):
    x1, x2 = y_pred
    y_pred = torch.cosine_similarity(x1, x2, dim=1)

    n_y = y_true.size(0)
    y_pred = torch.squeeze(y_pred)
    y_true = torch.squeeze(y_true)
    y_pred_thresholded = torch.zeros_like(y_pred)
    y_pred_thresholded[y_pred >= 0.5] = 1
    y_pred_thresholded[y_pred < 0.5] = -1

    acc = (y_pred_thresholded == y_true)
    acc = torch.sum(acc).item()
    acc = acc / float(n_y)
    return acc

def map(y_pred, y_true):
    """
    Returns mAP
    """
    y_pred = np.array(y_pred.tolist())
    y_true = np.array(y_true.tolist())

    n_classes = y_true.shape[1]
    map = [average_precision_score(y_true[:, i], y_pred[:, i]) for i in range(n_classes)]
    map = np.nan_to_num(map)
    map = np.mean(map)
    map = torch.tensor(map)
    return map

def recall_at_k(logits, labels, K=10):
    """
    Recall@K
    """
    B = labels.numel()
    assert logits.shape[0] == B, f"{logits.shape[0]} != {B}"
    sorted_idxs = logits.sort(dim=1, descending=True).indices
    recall_k = torch.zeros(K)
    for k in range(K):
        equals = sorted_idxs[:, :(k + 1)] == labels.view(B, 1)
        correct = equals.sum().item()
        recall_k[k] = correct
    return recall_k, B

class METRIC_FUNCTIONS:
    binary_siamese = binary_siamese
    binary_hinge = binary_hinge
    accuracy = accuracy
    binary_accuracy = binary_accuracy
    cosine_similarity_accuracy = cosine_similarity_accuracy
    recall_at_k = recall_at_k
    map = map
