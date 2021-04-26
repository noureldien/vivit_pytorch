"""
Code modified from:
https://raw.githubusercontent.com/lucidrains/vit-pytorch/main/vit_pytorch/vit.py
"""

import torch
from torch import nn, einsum, optim
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from core import consts, metrics, pytorch_utils


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim, out_dim),nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_patches_space=None, num_patches_time=None, attn_type=None):
        super().__init__()

        assert attn_type in ['space', 'time'], 'Attention type should be one of the following: space, time.'

        self.attn_type = attn_type
        self.num_patches_space = num_patches_space
        self.num_patches_time = num_patches_time

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    def forward(self, x):

        t = self.num_patches_time
        n = self.num_patches_space

        # reshape to reveal dimensions of space and time
        x = rearrange(x, 'b (t n) d -> b t n d', t=t, n=n)

        if self.attn_type == 'space':
            out = self.forward_space(x) # (b, tn, d)
        elif self.attn_type == 'time':
            out = self.forward_time(x) # (b, tn, d)
        else:
            raise Exception('Unknown attention type: %s' % (self.attn_type))

        return out

    def forward_space(self, x):
        """
        x: (b, t, n, d)
        """

        t = self.num_patches_time
        n = self.num_patches_space

        # hide time dimension into batch dimension
        x = rearrange(x, 'b t n d -> (b t) n d')  # (bt, n, d)

        # apply self-attention
        out = self.forward_attention(x)  # (bt, n, d)

        # recover time dimension and merge it into space
        out = rearrange(out, '(b t) n d -> b (t n) d', t=t, n=n)  # (b, tn, d)

        return out

    def forward_time(self, x):
        """
        x: (b, t, n, d)
        """

        t = self.num_patches_time
        n = self.num_patches_space

        # hide time dimension into batch dimension
        x = x.permute(0, 2, 1, 3)  # (b, n, t, d)
        x = rearrange(x, 'b n t d -> (b n) t d')  # (bn, t, d)

        # apply self-attention
        out = self.forward_attention(x)  # (bn, t, d)

        # recover time dimension and merge it into space
        out = rearrange(out, '(b n) t d -> b (t n) d', t=t, n=n)  # (b, tn, d)

        return out

    def forward_attention(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, num_patches_space, num_patches_time):
        super().__init__()

        self.num_patches_space = num_patches_space
        self.num_patches_time = num_patches_time
        heads_half = int(heads / 2.0)

        assert dim % 2 == 0

        self.attention_space = PreNorm(dim, Attention(dim, heads=heads_half, dim_head=dim_head, dropout=dropout, num_patches_space=num_patches_space, num_patches_time=num_patches_time, attn_type='space'))
        self.attention_time = PreNorm(dim, Attention(dim, heads=heads_half, dim_head=dim_head, dropout=dropout, num_patches_space=num_patches_space, num_patches_time=num_patches_time, attn_type='time'))

        inner_dim = dim_head * heads_half * 2
        self.linear = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.mlp = PreNorm(dim, FeedForward(dim, mlp_dim, dim, dropout=dropout))

    def forward(self, x):

        # self-attention
        xs = self.attention_space(x)
        xt = self.attention_time(x)
        out_att = torch.cat([xs, xt], dim=2)

        # linear after self-attention
        out_att = self.linear(out_att)

        # residual connection for self-attention
        out_att += x

        # mlp after attention
        out_mlp = self.mlp(out_att)

        # residual for mlp
        out_mlp += out_att

        return out_mlp


class ViViT(nn.Module):
    def __init__(self, num_classes, clip_size):
        super().__init__()

        self.image_size = 224
        self.patch_size = 28
        self.num_classes = num_classes
        self.clip_size = clip_size
        self.pool_type = 'cls'
        self.dim = 256
        self.depth = 6
        self.heads = 16
        self.mlp_dim = 256
        self.channels = 3
        self.dim_head = 64
        self.dropout_ratio = 0.1
        self.emb_dropout_ratio = 0.1

        assert self.heads % 2 == 0, 'Number of heads should be even.'

        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches_time = clip_size
        self.num_patches_space = (self.image_size // self.patch_size) ** 2
        self.num_patches = self.num_patches_time * self.num_patches_space

        self.patch_dim = self.channels * self.patch_size ** 2
        assert self.pool_type in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # init layers of the classifier
        self._init_layers()

        # init loss, metric, and optimizer
        self._loss_fn = nn.CrossEntropyLoss()
        self._metric_fn = metrics.accuracy
        # self._optimizer = optim.Adam(self.parameters(), 0.001)
        self._optimizer = optim.SGD(self.parameters(), 0.1)

    def _init_layers(self):

        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size), nn.Linear(self.patch_dim, self.dim), )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        self.dropout = nn.Dropout(self.emb_dropout_ratio)

        self.transformers = nn.ModuleList([])
        for _ in range(self.depth):
            self.transformers.append(Transformer(self.dim, self.heads, self.dim_head, self.mlp_dim, self.dropout_ratio, self.num_patches_space, self.num_patches_time))

        self.mlp_head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.num_classes), nn.Softmax())

    def forward(self, x):

        b, c, t, h, w = x.shape

        # hide time inside batch
        x = x.permute(0, 2, 1, 3, 4)  # (b, t, c, h, w)
        x = rearrange(x, 'b t c h w -> (b t) c h w')  # (b*t, c, h, w)

        # input embedding to get patch token
        x = self.to_patch_embedding(x)  # (b*t, n, d)

        # concat patch token and class token
        x = rearrange(x, '(b t) n d -> b (t n) d', b=b, t=t)  # (b, tn, d)

        # add position embedding
        x += self.pos_embedding  # (b, tn, d)
        x = self.dropout(x)  # (b, tn, d)

        # layers of transformers
        for transformer in self.transformers:
            x = transformer(x)  # (b, tn, d)

        # space-time pooling
        x = x.mean(dim=1)

        # classification
        # x = x[:, 0]

        # classifier
        x = self.mlp_head(x)
        return x
