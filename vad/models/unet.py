import torch
import torch.nn as nn


def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNetPredictor(nn.Module):
    """Baseline next-frame predictor: clip stacked on channels -> U-Net -> next frame in [0,1].
    Input H and W must be divisible by 8."""

    def __init__(self, clip_len=8, base=32):
        super().__init__()
        b = base
        self.enc1, self.enc2 = _block(3 * clip_len, b), _block(b, 2 * b)
        self.enc3, self.enc4 = _block(2 * b, 4 * b), _block(4 * b, 8 * b)
        self.pool = nn.MaxPool2d(2)
        self.up3, self.up2, self.up1 = (nn.ConvTranspose2d(8 * b, 4 * b, 2, 2),
                                        nn.ConvTranspose2d(4 * b, 2 * b, 2, 2),
                                        nn.ConvTranspose2d(2 * b, b, 2, 2))
        self.dec3, self.dec2, self.dec1 = _block(8 * b, 4 * b), _block(4 * b, 2 * b), _block(2 * b, b)
        self.out = nn.Conv2d(b, 3, 1)

    def forward(self, clip, flow=None):            # clip: B,T,C,H,W ; flow ignored by this baseline
        bsz, t, c, h, w = clip.shape
        x = clip.reshape(bsz, t * c, h, w)
        # Encoder: shrink the image step by step while collecting features. The decoder below upsamples and reuses them (skip connections).
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))
