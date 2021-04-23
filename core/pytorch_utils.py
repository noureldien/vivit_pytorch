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
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.onnx
from torch import nn
from torch.nn import functional as F
from torch import onnx
from torchviz import make_dot
from thop import profile

from modules import layers
from core import utils, data_utils, config_utils, consts
from core.utils import Path as Pth

# region Model Save/Load

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    return model

def save_model_dict(model, path):
    model_dict = model.state_dict()
    torch.save(model_dict, path)

def load_model_dict(model, path, strict=True, resolve_multi_gpu=False):
    model_dict = torch.load(path)
    model_dict = __model_dict_resolve_multi_gpu(model_dict) if resolve_multi_gpu else model_dict
    model.load_state_dict(model_dict, strict=strict)
    return model

def __model_dict_resolve_multi_gpu(model_dict):
    updated_dict = dict()
    for k, v in model_dict.items():
        k = k.replace('module.', '')
        updated_dict[k] = v

    return updated_dict

# endregion

# region Model Summary/Visualization

def model_summary(model, input_size, batch_size=-1, device="cuda", logger=consts.CONSOLE_LOGGER):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # logger.print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # logger.print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    logger.print("-------------------------------------------------------------------------------------------------------")
    line_new = "{:>50}  {:>30} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    logger.print(line_new)
    logger.print("=======================================================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>50}  {:>30} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        logger.print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    logger.print("=======================================================================================================")
    logger.print("Total params: {0:,}".format(total_params))
    logger.print("Trainable params: {0:,}".format(trainable_params))
    logger.print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    logger.print("-------------------------------------------------------------------------------------------------------")
    logger.print("Input size (MB): %0.2f" % total_input_size)
    logger.print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    logger.print("Params size (MB): %0.2f" % total_params_size)
    logger.print("Estimated Total Size (MB): %0.2f" % total_size)
    logger.print("-------------------------------------------------------------------------------------------------------")

def model_summary_multi_input(model, input_sizes, batch_size=-1, device="cuda", logger=utils.TextLogger(consts.LOGGING_TYPES.console)):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_sizes]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # logger.print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    logger.print("-------------------------------------------------------------------------------------------------------")
    line_new = "{:>50}  {:>30} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    logger.print(line_new)
    logger.print("=======================================================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>50}  {:>30} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        logger.print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = sum([abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.)) for input_size in input_sizes])
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    logger.print("=======================================================================================================")
    logger.print("Total params: {0:,}".format(total_params))
    logger.print("Trainable params: {0:,}".format(trainable_params))
    logger.print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    logger.print("-------------------------------------------------------------------------------------------------------")
    logger.print("Input size (MB): %0.2f" % total_input_size)
    logger.print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    logger.print("Params size (MB): %0.2f" % total_params_size)
    logger.print("Estimated Total Size (MB): %0.2f" % total_size)
    logger.print("-------------------------------------------------------------------------------------------------------")

def visualize_model(model, input_shape):
    # add batch size
    input_shape = [2] + list(input_shape)

    # create input tensor
    input_tensor = torch.randn(input_shape)
    input_tensor = torch.autograd.Variable(input_tensor).cuda()

    # feedforward to get output tensor
    output_tensor = model(input_tensor)

    # get model params
    model_params = dict(model.named_parameters())

    # plot graph
    g = make_dot(output_tensor.mean(), params=model_params)

    g.view()

def export_model_definition(model, input_shape, model_path='model.onnx'):
    # add batch size
    input_shape = [2] + list(input_shape)

    # create input tensor
    input_tensor = torch.randn(input_shape)
    input_tensor = torch.autograd.Variable(input_tensor).cuda()

    torch.onnx.export(model, input_tensor, model_path)

def calc_model_params(model, mode='MB'):
    KB = 1000.0
    MB = 1000 * KB
    GB = 1000 * MB
    parms = sum(p.numel() for p in model.parameters())

    if mode == 'GB':
        parms /= GB
    elif mode == 'MB':
        parms /= MB
    elif mode == 'KB':
        parms /= KB
    else:
        raise Exception('Sorry, unsupported mode: %s' % (mode))

    return parms

def calc_model_flops(model, batch_shape, mode='M', verbose=True):
    input = torch.randn(*batch_shape).cuda()
    flops, _ = profile(model, (input,), verbose=verbose)
    if mode is None:
        pass
    elif mode == 'K':
        flops /= (1024.0)
    elif mode == 'M':
        flops /= (1024.0 * 1024.0)
    elif mode == 'G':
        flops /= (1024.0 * 1024.0 * 1024.0)
    return flops

def convert_pytorch_to_onnx(input_shape, weight_path, model_class, output_name):
    # A model class instance (class not shown)
    model = model_class()

    # Load the weights from a file (.pth usually)
    state_dict = torch.load(weight_path)

    # Load the weights now into a model net architecture defined by our class
    model.load_state_dict(state_dict)

    # Create the right input shape (e.g. for an image)
    input = torch.randn(*input_shape)

    output_name = '%s.onnx' % (output_name)
    torch.onnx.export(model, input, output_name)

# endregion

# region Freeze/Unfreeze

def freeze_layer_recursive(layer):
    for param in layer.parameters():
        param.requires_grad = False

    sub_layers = layer.children()
    freeze_layers_recursive(sub_layers)

def freeze_layers(layers):
    # unfreeze given layers
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False

def freeze_layers_recursive(layers):
    [freeze_layer_recursive(l) for l in layers]

def freeze_model_layers(model, layer_names):
    # freeze given layers
    layers = [getattr(model, l_name) for l_name in layer_names]
    freeze_layers(layers)

def freeze_model_layers_recursive(model, layer_names):
    # freeze given layers
    layers = [getattr(model, l_name) for l_name in layer_names]
    freeze_layers_recursive(layers)

def freeze_model(model):
    params = model.parameters()
    for param in params:
        param.requires_grad = False

def unfreeze_layer_recursive(layer):
    for param in layer.parameters():
        param.requires_grad = True

    sub_layers = layer.children()
    unfreeze_layers_recursive(sub_layers)

def unfreeze_layers(layers):
    # unfreeze given layers
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = True

def unfreeze_layers_recursive(layers):
    [unfreeze_layer_recursive(l) for l in layers]

def unfreeze_model_layers(model, layer_names):
    # unfreeze given layers
    layers = [getattr(model, l_name) for l_name in layer_names]
    unfreeze_layers(layers)

def unfreeze_model_layers_recursive(model, layer_names):
    # unfreeze given layers
    layers = [getattr(model, l_name) for l_name in layer_names]
    unfreeze_layers_recursive(layers)

# endregion

# region Learning Rate

def get_learning_rates(optimizer, n_gpus=1):
    lr = [g['lr'] * n_gpus for g in optimizer.param_groups]
    return lr

def update_learning_rates(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

# endregion

# region Misc

def configure_specific_gpu():
    _is_local_machine = config_utils.is_local_machine()

    if _is_local_machine:
        gpu_core_id = 0
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--gpu_core_id', default='-1', type=int)
        args = parser.parse_args()
        gpu_core_id = args.gpu_core_id

        if gpu_core_id < 0 or gpu_core_id > 3:
            msg = 'Please specify a correct GPU core!!!'
            raise Exception(msg)

    torch.cuda.set_device(gpu_core_id)

    # set which device to be used
    consts.GPU_CORE_ID = gpu_core_id

def get_shape(tensor):
    t_shape = list(tensor.shape)
    return t_shape

def print_shape(tensor):
    t_shape = get_shape(tensor)
    print(t_shape)

def parallelize_model(model, gpu_ids):
    n_gpus = len(gpu_ids)
    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        if hasattr(model, 'module'):
            model._optimizer = model.module._optimizer
            model._scheduler = model.module._scheduler
            model._metric_fn = model.module._metric_fn
            model._loss_fn = model.module._loss_fn
    return model

# endregion

# region classes: Optimization

class Optimization():
    def __init__(self):
        optimizer = None
        lr = None

    def on_epoch_ends(self):
        # update lr rate
        pass

# endregion

# region Classes: Callbacks

class BaseCallback(object):
    def __init__(self, model):
        self.model = model

    def on_batch_ends(self, batch_num, is_training):
        pass

    def on_epoch_ends(self, epoch_num):
        pass

class ModelSaveCallback(BaseCallback):
    def __init__(self, model, model_root_path, frequency=1, resolve_multi_gpu=True):
        super(ModelSaveCallback, self).__init__(model)
        self.model = model
        self.frequency = frequency
        self.resolve_multi_gpu = resolve_multi_gpu
        self.model_root_path = model_root_path

        utils.make_directory(model_root_path)

    def on_batch_ends(self, batch_num, is_training):
        pass

    def on_epoch_ends(self, epoch_num):
        if epoch_num % self.frequency != 0:
            return

        model_root_path = self.model_root_path
        model_dict_path = '%s/%03d.pt' % (model_root_path, epoch_num)

        # if multi gpu, then save only the single-gpu model
        model = self.model.module if self.resolve_multi_gpu and hasattr(self.model, 'module') else self.model

        # save both model and model dict
        # save_model(model, model_path)
        save_model_dict(model, model_dict_path)

# endregion
