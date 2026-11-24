"""Spatial branch: a small Vision Transformer over patch tokens (written from scratch so attention maps
are exposed for localization in Phase 7, and no pretrained-weight download is needed yet)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads, attn_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.h, self.drop = heads, attn_drop
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.keep_attn, self.last_attn = False, None

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if self.keep_attn:                                   # explicit softmax so the map can be stored
            a = ((q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5).softmax(-1)
            self.last_attn = a.detach()
            out = a @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class SpatialBranch(nn.Module):
    """[B, 3*in_frames, H, W] -> tokens [B, N, D], N = (H/patch)^2. Self-attention over patches models
    crowd density and interactions."""

    def __init__(self, img_size=224, patch=16, in_frames=8, embed_dim=256, depth=6, heads=8, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch == 0
        self.img_size, self.grid = img_size, img_size // patch
        self.patch_embed = nn.Conv2d(3 * in_frames, embed_dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        assert x.shape[-1] == self.img_size and x.shape[-2] == self.img_size, "input size must equal img_size"
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)

    def set_keep_attn(self, flag: bool) -> None:
        for blk in self.blocks:
            blk.attn.keep_attn = flag

    def attention_maps(self):
        """List (one per block) of [B, heads, N, N] from the last forward pass with keep_attn on."""
        return [blk.attn.last_attn for blk in self.blocks]


class CrossAttention(nn.Module):
    """Queries come from x, keys/values from ctx. Used for appearance<->motion fusion and temporal pooling."""

    def __init__(self, dim, heads, attn_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.h, self.drop = heads, attn_drop
        self.q, self.kv, self.proj = nn.Linear(dim, dim), nn.Linear(dim, dim * 2), nn.Linear(dim, dim)
        self.keep_attn, self.last_attn = False, None

    def forward(self, x, ctx):
        b, n, d = x.shape
        m = ctx.shape[1]
        q = self.q(x).reshape(b, n, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(ctx).reshape(b, m, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if self.keep_attn:
            a = ((q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5).softmax(-1)
            self.last_attn = a.detach()
            out = a @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))
