import math

import torch
import torch.nn as nn

from vad.models.vit import SpatialBranch


class SpatialPredictor(nn.Module):
    """Spatial-branch-only next-frame predictor (Phase 3 ablation / stepping stone).

    The most recent `spatial_frames` frames are stacked in the patch embedding (early fusion over time), a ViT
    models patch interactions, and a conv decoder upsamples the token grid back to pixels. With residual=True the
    network predicts a CHANGE relative to the last input frame; its output layer is zero-initialised, so training
    starts from the 'copy the last frame' predictor."""

    def __init__(self, img_size=224, patch=16, clip_len=8, spatial_frames=8, embed_dim=256, depth=6, heads=8,
                 mlp_ratio=4.0, residual=True):
        super().__init__()
        self.k, self.residual = min(spatial_frames, clip_len), residual
        self.branch = SpatialBranch(img_size, patch, self.k, embed_dim, depth, heads, mlp_ratio)
        stages = int(math.log2(patch))
        assert 2 ** stages == patch, "patch must be a power of two"
        chs = [embed_dim] + [max(16, 128 // 2 ** i) for i in range(stages)]
        layers = []
        for i in range(stages):
            layers += [nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2), nn.BatchNorm2d(chs[i + 1]), nn.ReLU(inplace=True),
                       nn.Conv2d(chs[i + 1], chs[i + 1], 3, padding=1), nn.BatchNorm2d(chs[i + 1]), nn.ReLU(inplace=True)]
        self.decoder = nn.Sequential(*layers)
        self.out = nn.Conv2d(chs[-1], 3, 1)
        if residual:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, clip, flow=None):               # clip: B,T,C,H,W ; flow unused until Phase 4
        b, t, c, h, w = clip.shape
        # Stack the latest frames, cut them into patches, and let the Transformer relate the patches to each other.
        tokens = self.branch(clip[:, -self.k:].reshape(b, self.k * c, h, w))      # B,N,D
        g = self.branch.grid
        y = self.out(self.decoder(tokens.transpose(1, 2).reshape(b, -1, g, g)))
        return clip[:, -1] + y if self.residual else torch.sigmoid(y)
