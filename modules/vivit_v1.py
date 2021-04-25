"""
Code borrowed from
https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py
"""

import numpy as np

import torch
from torch import nn, einsum, optim
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from core import consts, metrics, pytorch_utils

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(scale_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                     )

        self.depth_conv = nn.Sequential(nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
                                        nn.BatchNorm2d(scale_dim),
                                        nn.GELU(),
                                        Rearrange('b c h w -> b (h w) c', h=14, w=14)
                                        )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x

class LCAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViViT(nn.Module):

    def __init__(self, n_classes, clip_size, is_train):
        super(ViViT, self).__init__()

        self.is_train = is_train
        self.clip_size = clip_size
        self.n_classes = n_classes

        # init layers of the classifier
        self._init_layers()

        # init loss, metric, and optimizer
        self._loss_fn = nn.CrossEntropyLoss()
        self._metric_fn = metrics.accuracy
        self._optimizer = optim.Adam(self.parameters(), 0.01)

    def _init_layers(self):

        # get configs of the model
        clip_size = self.clip_size
        n_classes = self.n_classes
        self.pool = 'cls' # either 'cls' or 'mean'
        image_size = 224
        image_patch_size = 16
        dim = 192
        depth = 4
        heads = 3

        in_channels = 3
        dim_head = 64
        dropout = 0.0
        emb_dropout = 0.0
        scale_dim = 4

        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % image_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        n_patches = (image_size // image_patch_size) ** 2
        patch_dim = in_channels * image_patch_size ** 2
        self.to_patch_embedding = nn.Sequential(Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=image_patch_size, p2=image_patch_size), nn.Linear(patch_dim, dim), )

        # positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, clip_size, n_patches + 1, dim))

        # space transformer
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        # time transformer
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        # mlp classifier
        self.dropout = nn.Dropout(emb_dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes), nn.Softmax())

    def forward(self, x):
        """
        Input shape: (b, c, t, h, w)
        b: batch size
        t: temporal dimensions, i.e. number of frames
        c: number of channels = 3
        h, w: height and width of the frame
        n: number of patches in the image = 196
        d: feature dimension after patch embedding = 192
        """

        x = rearrange(x, 'b c t h w -> b t c h w') # (b, t, c, h, w)
        x = self.to_patch_embedding(x)  # (b, t, n, h, w)
        b, t, n, d = x.shape # (b, t, n)

        # batch encoding
        # class token econding
        # position encoding
        # 4. space transformer

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t) # (b, t, 1, d)
        x = torch.cat((cls_space_tokens, x), dim=2) # (b, t, n+1, d)
        x += self.pos_embedding[:, :, :(n + 1)] # (b, t, n+1, d)
        x = self.dropout(x) # (b, t, n+1, d)

        # 4. spatial transformer
        x = rearrange(x, 'b t n d -> (b t) n d') # (b*t, n+1, d)
        x = self.space_transformer(x) # (b*t, n+1, d)

        # 5. spatial pooling
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b) # (b, 1, d)
        x = torch.cat((cls_temporal_tokens, x), dim=1) # (b, t+1, d)

        x = self.temporal_transformer(x) # (b, t+1, d)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)

        return x