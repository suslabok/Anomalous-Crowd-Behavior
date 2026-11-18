"""Random clips so the training skeleton can be smoke-tested before real data (Phase 1)."""
import torch
from torch.utils.data import Dataset


class DummyClips(Dataset):
    def __init__(self, n=64, clip_len=8, size=64):
        self.n, self.t, self.s = n, clip_len, size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        clip = torch.rand(self.t, 3, self.s, self.s, generator=g)
        target = torch.rand(3, self.s, self.s, generator=g)
        return {"clip": clip, "target": target}
