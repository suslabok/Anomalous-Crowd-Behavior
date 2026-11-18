import torch
import torch.nn as nn


class DummyPredictor(nn.Module):
    """Placeholder: stacks the clip along channels and predicts the next frame."""

    def __init__(self, clip_len=8, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 * clip_len, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )

    def forward(self, clip):                      # clip: B,T,C,H,W
        b, t, c, h, w = clip.shape
        return self.net(clip.reshape(b, t * c, h, w))
