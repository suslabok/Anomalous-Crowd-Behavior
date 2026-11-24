"""Temporal branch: divided space-time attention over per-step motion tokens.

Input is a sequence of motion fields (optical flow and/or frame differences), one per consecutive frame pair.
Each block attends across TIME at every patch location (trajectories, velocity changes), then across SPACE within
every time step (neighbouring motion: counter-flow, sudden dispersal). Attention pooling over time gives one
motion token per patch location."""
import torch
import torch.nn as nn

from vad.models.vit import Attention, CrossAttention


class DividedSTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.nt, self.ns, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn_t, self.attn_s = Attention(dim, heads), Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):                                   # x: [B, T, N, D]
        b, t, n, d = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(b * n, t, d)     # sequences over time, one per location
        xt = xt + self.attn_t(self.nt(xt))
        x = xt.reshape(b, n, t, d).permute(0, 2, 1, 3)
        xs = x.reshape(b * t, n, d)                         # sequences over space, one per time step
        xs = xs + self.attn_s(self.ns(xs))
        x = xs.reshape(b, t, n, d)
        return x + self.mlp(self.n2(x))


class TemporalBranch(nn.Module):
    """[B, T, C, H, W] motion fields -> motion tokens [B, N, D], N = (H/patch)^2."""

    def __init__(self, img_size=224, patch=16, in_ch=2, steps=7, embed_dim=256, depth=2, heads=8, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch == 0
        self.img_size, self.grid, self.steps = img_size, img_size // patch, steps
        n = self.grid ** 2
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, patch, patch)      # shared across time steps
        self.pos = nn.Parameter(torch.zeros(1, 1, n, embed_dim))
        self.tpos = nn.Parameter(torch.zeros(1, steps, 1, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.tpos, std=0.02)
        self.blocks = nn.ModuleList([DividedSTBlock(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.pool_q = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pool_q, std=0.02)
        self.pool = CrossAttention(embed_dim, heads)
        self.pool_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, t, c, h, w = x.shape
        assert t == self.steps, f"expected {self.steps} time steps, got {t}"
        assert h == self.img_size and w == self.img_size, "input size must equal img_size"
        n = self.grid ** 2
        tok = self.patch_embed(x.reshape(b * t, c, h, w)).flatten(2).transpose(1, 2).reshape(b, t, n, -1)
        tok = tok + self.pos + self.tpos
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        kv = tok.permute(0, 2, 1, 3).reshape(b * n, t, -1)                # per location: its T tokens
        q = self.pool_q.expand(b * n, 1, -1)
        return self.pool_norm(self.pool(q, kv).reshape(b, n, -1))
