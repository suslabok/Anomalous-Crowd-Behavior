import math

import torch.nn as nn


def init_head(conv, mode):
    """Output-layer init for residual models. 'zero' starts exactly at 'copy the last frame' (but gives the rest of
    the network no gradient until the head has moved); 'small' starts near it with gradients flowing from step 0."""
    if mode == "zero":
        nn.init.zeros_(conv.weight)
    elif mode == "small":
        nn.init.normal_(conv.weight, std=0.01)
    else:
        raise ValueError("head_init must be 'zero' or 'small'")
    nn.init.zeros_(conv.bias)


class TokenDecoder(nn.Module):
    """Token grid [B, N, D] -> pixel tensor [B, 3, H, W] by log2(patch) conv-transpose upsampling stages."""

    def __init__(self, embed_dim, patch, zero_init=True, head_init="zero"):
        super().__init__()
        stages = int(math.log2(patch))
        assert 2 ** stages == patch, "patch must be a power of two"
        chs = [embed_dim] + [max(16, 128 // 2 ** i) for i in range(stages)]
        layers = []
        for i in range(stages):
            layers += [nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2), nn.BatchNorm2d(chs[i + 1]), nn.ReLU(inplace=True),
                       nn.Conv2d(chs[i + 1], chs[i + 1], 3, padding=1), nn.BatchNorm2d(chs[i + 1]), nn.ReLU(inplace=True)]
        self.decoder = nn.Sequential(*layers)
        self.out = nn.Conv2d(chs[-1], 3, 1)
        if zero_init:
            init_head(self.out, head_init)

    def forward(self, tokens, grid):
        b = tokens.shape[0]
        return self.out(self.decoder(tokens.transpose(1, 2).reshape(b, -1, grid, grid)))
