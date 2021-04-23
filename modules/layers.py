# coding=utf-8

import math
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from torch import distributions
from torch.autograd import Variable
from torchvision import datasets, transforms

from core import pytorch_utils

# region Basic Layers

class Max(nn.Module):
    def __init__(self, dim, keepdim=False):
        super(Max, self).__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        dim_type = type(self.dim)
        dims = self.dim if (dim_type is list or dim_type is tuple) else [self.dim]
        dims = np.sort(dims)[::-1]
        dims = [int(d) for d in dims]
        tensor = input
        for d in dims:
            tensor, _ = torch.max(tensor, dim=d, keepdim=self.keepdim)
        return tensor

class Mean(nn.Module):
    def __init__(self, dim, keepdim=False):
        super(Mean, self).__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        dim_type = type(self.dim)
        dims = self.dim if (dim_type is list or dim_type is tuple) else [self.dim]
        dims = np.sort(dims)[::-1]
        dims = [int(d) for d in dims]
        tensor = input
        for d in dims:
            tensor = torch.mean(tensor, dim=d, keepdim=self.keepdim)
        return tensor

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        pass

    def forward(self, input):
        batch_size = input.size(0)
        output = input.view(batch_size, -1)
        return output

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim
        pass

    def forward(self, input):

        if self.dim is None:
            output = torch.squeeze(input)
        else:
            dim_type = type(self.dim)
            dims = self.dim if (dim_type is list or dim_type is tuple) else [self.dim]

            output = input
            for d in dims:
                output = torch.squeeze(output, dim=d)

        return output

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
        pass

    def forward(self, input):
        B = pytorch_utils.get_shape(input)[0]
        new_shape = [B] + list(self.shape)
        output = input.view(*new_shape)

        return output

class ReshapeInduced(nn.Module):
    def __init__(self, shape):
        super(ReshapeInduced, self).__init__()
        self.shape = shape
        pass

    def forward(self, input):
        new_shape = [-1] + list(self.shape)
        output = input.view(*new_shape)

        return output

class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims
        pass

    def forward(self, input):
        output = input.permute(*self.dims)

        return output

class Normalize(nn.Module):
    def __init__(self, dim):
        super(Normalize, self).__init__()
        self.dim = dim
        pass

    def forward(self, input):
        output = F.normalize(input, dim=self.dim)
        return output

# endregion

# region BatchNorm with Axis

class BatchNorm(nn.Module):
    """
    Batch norm of features for linear layer, with ability to specify axis.
    """

    def __init__(self, num_features, dim=1):
        super(BatchNorm, self).__init__()

        assert dim in [0, 1]

        self.dim = dim
        self.num_features = num_features
        self.layer = nn.BatchNorm1d(num_features)

    def forward(self, input):

        input_shape = pytorch_utils.get_shape(input)
        assert len(input_shape) == 2

        dim = self.dim

        # permute to put the required dimension in the 2nd dimension
        if dim == 0:
            x = input.permute(1, 0)
        else:
            x = input

        # apply batch_norm
        num_features = pytorch_utils.get_shape(x)[1]
        assert num_features == self.num_features
        x = self.layer(x)

        # permute back to the original view
        if dim == 0:
            x = x.permute(1, 0)

        x_shape = pytorch_utils.get_shape(x)
        assert input_shape == x_shape

        return x

class BatchNorm1d(nn.Module):

    def __init__(self, num_features, dim=1):
        super(BatchNorm1d, self).__init__()

        assert dim in [1, 2]

        self.dim = dim
        self.num_features = num_features
        self.layer = nn.BatchNorm1d(num_features)

    def forward(self, input):

        input_shape = pytorch_utils.get_shape(input)
        assert len(input_shape) == 3

        dim = self.dim

        # permute to put the required dimension in the 2nd dimension
        if dim == 1:
            x = input
        elif dim == 2:
            x = input.permute(0, 2, 1)

        # apply batch_norm
        num_features = pytorch_utils.get_shape(x)[1]
        assert num_features == self.num_features
        x = self.layer(x)

        # permute back to the original view
        if dim == 2:
            x = x.permute(0, 2, 1)

        x_shape = pytorch_utils.get_shape(x)
        assert input_shape == x_shape

        return x

class BatchNorm2d(nn.Module):

    def __init__(self, num_features, dim=1):
        super(BatchNorm2d, self).__init__()

        assert dim in [1, 2, 3]

        self.dim = dim
        self.num_features = num_features
        self.layer = nn.BatchNorm3d(num_features)

    def forward(self, input):

        input_shape = pytorch_utils.get_shape(input)
        assert len(input_shape) == 4

        dim = self.dim

        # permute to put the required dimension in the 2nd dimension
        if dim == 1:
            x = input
        elif dim == 2:
            x = input.permute(0, 2, 3, 1)
        elif dim == 3:
            x = input.permute(0, 1, 3, 2)

        # apply batch_norm
        num_features = pytorch_utils.get_shape(x)[1]
        assert num_features == self.num_features
        x = self.layer(x)

        # permute back to the original view
        if dim == 2:
            x = x.permute(0, 2, 3, 1)
        elif dim == 3:
            x = x.permute(0, 1, 3, 2)

        x_shape = pytorch_utils.get_shape(x)
        assert input_shape == x_shape

        return x

class BatchNorm3d(nn.Module):

    def __init__(self, num_features, dim=1):
        super(BatchNorm3d, self).__init__()

        assert dim in [1, 2, 3, 4]

        self.dim = dim
        self.num_features = num_features
        self.layer = nn.BatchNorm3d(num_features)

    def forward(self, input):

        input_shape = pytorch_utils.get_shape(input)
        assert len(input_shape) == 5

        dim = self.dim

        # permute to put the required dimension in the 2nd dimension
        if dim == 1:
            x = input
        elif dim == 2:
            x = input.permute(0, 2, 1, 3, 4)
        elif dim == 3:
            x = input.permute(0, 3, 2, 1, 4)
        elif dim == 4:
            x = input.permute(0, 4, 2, 3, 1)

        # apply batch_norm
        num_features = pytorch_utils.get_shape(x)[1]
        assert num_features == self.num_features
        x = self.layer(x)

        # permute back to the original view
        if dim == 2:
            x = x.permute(0, 2, 1, 3, 4)
        elif dim == 3:
            x = x.permute(0, 3, 2, 1, 4)
        elif dim == 4:
            x = x.permute(0, 4, 2, 3, 1)

        x_shape = pytorch_utils.get_shape(x)
        assert input_shape == x_shape

        return x

# endregion

# region LayerNorm with Axis

class LayerNorm3d(nn.Module):

    def __init__(self, num_features, dim=1):
        super(LayerNorm3d, self).__init__()

        assert dim in [1, 2, 3, 4]

        self.dim = dim
        self.num_features = num_features
        self.layer = nn.LayerNorm(num_features)

    def forward(self, input):

        input_shape = pytorch_utils.get_shape(input)
        assert len(input_shape) == 5

        dim = self.dim

        # permute to put the required dimension in the 2nd dimension
        if dim == 4:
            x = input
        elif dim == 3:
            x = input.permute(0, 1, 2, 4, 3)
        elif dim == 2:
            x = input.permute(0, 1, 4, 3, 2)
        elif dim == 1:
            x = input.permute(0, 4, 2, 3, 1)
        else:
            x = None

        B, d1, d2, d3, d4 = pytorch_utils.get_shape(x)
        assert d4 == self.num_features

        # reshape
        x = x.view(B, d1 * d2 * d3, d4)

        # apply layer_norm
        x = self.layer(x)

        # reshape back to the original view
        x = x.view(B, d1, d2, d3, d4)

        # permute back to the original view
        if dim == 3:
            x = x.permute(0, 1, 2, 4, 3)
        elif dim == 2:
            x = x.permute(0, 1, 4, 3, 2)
        elif dim == 1:
            x = x.permute(0, 4, 2, 3, 1)

        x_shape = pytorch_utils.get_shape(x)
        assert input_shape == x_shape

        return x

# endregion

# region Linear 1D, 2D, 3D

class Linear1d(nn.Module):

    def __init__(self, in_features, out_features, dim=1):
        super(Linear1d, self).__init__()

        message = 'Sorry, unsupported dimension for linear layer: %d' % (dim)
        assert dim in [1, 2], message

        # weather to permute the dimesions or not
        self.is_permute = dim in [2]

        # permutation according to the dim
        if dim == 2:
            permutation_in = (0, 2, 1)
            permutation_out = (0, 2, 1)
        else:
            permutation_in = None
            permutation_out = None

        self.permutation_in = permutation_in
        self.permutation_out = permutation_out
        kernel_size = 1

        self.layer = nn.Conv1d(in_features, out_features, kernel_size)

    def forward(self, x_in):

        # in permutation
        x_out = x_in.permute(self.permutation_in) if self.is_permute else x_in

        # linear layer
        x_out = self.layer(x_out)

        # out permutation
        x_out = x_out.permute(self.permutation_out) if self.is_permute else x_out

        return x_out

class Linear2d(nn.Module):

    def __init__(self, in_features, out_features, dim=1, **kwargs):
        super(Linear2d, self).__init__()

        message = 'Sorry, unsupported dimension for linear layer: %d' % (dim)
        assert dim in [1, 2, 3], message

        # weather to permute the dimesions or not
        self.is_permute = dim in [2, 3]

        # permutation according to the dim
        if dim == 2:
            permutation_in = (0, 2, 1, 3)
            permutation_out = (0, 2, 1, 3)
        elif dim == 3:
            permutation_in = (0, 3, 2, 1)
            permutation_out = (0, 3, 2, 1)
        else:
            permutation_in = None
            permutation_out = None

        self.permutation_in = permutation_in
        self.permutation_out = permutation_out
        kernel_size = (1, 1)

        self.layer = nn.Conv2d(in_features, out_features, kernel_size, **kwargs)

    def forward(self, x_in):

        # in permutation
        x_out = x_in.permute(self.permutation_in) if self.is_permute else x_in

        # linear layer
        x_out = self.layer(x_out)

        # out permutation
        x_out = x_out.permute(self.permutation_out) if self.is_permute else x_out

        return x_out

class Linear3d(nn.Module):

    def __init__(self, in_features, out_features, dim=1, **kwargs):
        super(Linear3d, self).__init__()

        message = 'Sorry, unsupported dimension for linear layer: %d' % (dim)
        assert dim in [1, 2, 3, 4], message

        # weather to permute the dimesions or not
        self.is_permute = dim in [2, 3, 4]

        # permutation according to the dim
        if dim == 2:
            permutation_in = (0, 2, 1, 3, 4)
            permutation_out = (0, 2, 1, 3, 4)
        elif dim == 3:
            permutation_in = (0, 3, 2, 1, 4)
            permutation_out = (0, 3, 2, 1, 4)
        elif dim == 4:
            permutation_in = (0, 4, 2, 3, 1)
            permutation_out = (0, 4, 2, 3, 1)
        else:
            permutation_in = None
            permutation_out = None

        self.permutation_in = permutation_in
        self.permutation_out = permutation_out
        kernel_size = (1, 1, 1)

        self.layer = nn.Conv3d(in_features, out_features, kernel_size, **kwargs)

    def forward(self, x_in):

        # in permutation
        x_out = x_in.permute(self.permutation_in) if self.is_permute else x_in

        # linear layer
        x_out = self.layer(x_out)

        # out permutation
        x_out = x_out.permute(self.permutation_out) if self.is_permute else x_out

        return x_out

# endregion
