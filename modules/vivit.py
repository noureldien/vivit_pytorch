"""
Code borrowed from
https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py
"""

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch import optim

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from modules.transformer import Transformer
from core import consts, metrics, pytorch_utils

class ViViT(nn.Module):

    def __init__(self, n_frames, is_train):
        super(ViViT, self).__init__()

        self.is_train = is_train
        self.n_frames = n_frames

        # init layers of the classifier
        self._init_layers()

        self._init_optimizer()
        self._init_loss()

    def _init_loss(self):

        # init loss, metric, scheduler
        self._loss_fn = nn.CrossEntropyLoss()
        self._metric_fn = metrics.accuracy

    def _init_optimizer(self):
        learning_rate = 0.01
        self._optimizer = optim.Adam(self.parameters(), learning_rate)

    def _init_layers(self):

        # get configs of the model
        n_frames = self.n_frames
        n_classes = 5
        n_classes = 174
        image_size_resize = 256
        image_size_crop = 224
        image_patch_size = 16
        dim = 192
        depth = 4
        heads = 3
        pool = 'cls' # either 'cls' or 'mean'
        in_channels = 3
        dim_head = 64
        dropout = 0.0
        emb_dropout = 0.0
        scale_dim = 4

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size_crop % image_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        n_patches = (image_size_crop // image_patch_size) ** 2
        patch_dim = in_channels * image_patch_size ** 2
        self.to_patch_embedding = nn.Sequential(Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=image_patch_size, p2=image_patch_size), nn.Linear(patch_dim, dim), )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_frames, n_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))

    def forward(self, x):
        """
        Input shape B, C, T, H, W
        B: Batch size
        C: Image channels (C=3)
        H: Image Height
        W: Image Width
        """

        B, T, C, H, W = pytorch_utils.get_shape(x)

        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape # batch size, time, n_image_patches,

        # batch encoding
        # class token econding
        # position encoding

        # space trans

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)

        return x