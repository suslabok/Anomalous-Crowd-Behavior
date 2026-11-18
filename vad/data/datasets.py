"""Torch datasets on top of VideoStore."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from vad.data.clips import VideoStore
from vad.data.common import list_videos, split_train_val, train_targets


def _tensors(d: dict) -> dict:
    return {k: torch.as_tensor(v) for k, v in d.items()}


class TrainClips(Dataset):
    """Normal-only training clips: input = frames [t-L, t-1], target = frame t."""

    def __init__(self, processed_root, dataset, video_ids, clip_len, img_size, target_stride=1,
                 flow_mode="none", flow_root=None, flow_size=128, split="train"):
        self.stores = [VideoStore(processed_root, dataset, split, v, clip_len, img_size,
                                  flow_mode, flow_root, flow_size) for v in video_ids]
        self.index = [(i, int(t)) for i, s in enumerate(self.stores)
                      for t in train_targets(s.num_frames, clip_len, target_stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        return _tensors(self.stores[i].get_clip(t))


class EvalVideo(Dataset):
    """Every frame of one test video as a target (replicate-padded start), with its label."""

    def __init__(self, processed_root, dataset, video_id, clip_len, img_size,
                 flow_mode="none", flow_root=None, flow_size=128, split="test"):
        self.store = VideoStore(processed_root, dataset, split, video_id, clip_len, img_size,
                                flow_mode, flow_root, flow_size)
        self.labels = self.store.labels()
        self.meta = self.store.meta

    def __len__(self):
        return self.store.num_frames

    def __getitem__(self, t):
        d = self.store.get_clip(t)
        if self.labels is not None:
            d["label"] = np.int64(self.labels[t])
        return _tensors(d)


def _kw(cfg, paths):
    dc = cfg.data
    return dict(clip_len=dc.clip_len, img_size=dc.img_size, flow_mode=dc.flow_mode,
                flow_root=paths.flow, flow_size=dc.flow_size)


def build_train_val(cfg, paths, dataset=None):
    """(train_ds, val_ds, val_video_ids). Val = held-out NORMAL videos."""
    dc, name = cfg.data, dataset or cfg.data.dataset
    vids = list_videos(paths.processed, name, "train")
    if not vids:
        raise FileNotFoundError(
            f"No processed '{name}' training videos in {paths.processed}. "
            f"Run `python -m vad.data.prepare unpack {name}` (or `prepare`) first.")
    tr, va = split_train_val(vids, dc.val_fraction)
    kw = _kw(cfg, paths)
    train = TrainClips(paths.processed, name, tr, target_stride=dc.target_stride, **kw)
    val = TrainClips(paths.processed, name, va, target_stride=max(2, dc.target_stride), **kw) if va else None
    return train, val, va


def build_eval_videos(cfg, paths, dataset=None, split="test"):
    """{video_id: EvalVideo} for a whole dataset split."""
    name = dataset or cfg.data.dataset
    kw = _kw(cfg, paths)
    return {v: EvalVideo(paths.processed, name, v, split=split, **kw)
            for v in list_videos(paths.processed, name, split)}
