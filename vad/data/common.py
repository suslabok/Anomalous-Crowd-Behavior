"""Common on-disk layout for every dataset (numpy-only, no torch).

processed/<dataset>/<split>/<video_id>/
    meta.json            VideoMeta
    frames/000000.jpg    3-channel frames resized to STORE_SIZE x STORE_SIZE
    labels.npy           uint8 [T]        frame-level labels      (test split)
    masks.npz['masks']   uint8 [T,S,S]    pixel-level masks       (test split, if available)
flow/<dataset>/<split>/<video_id>.npy   uint8 [T-1,2,F,F] quantized Farneback flow
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

STORE_SIZE = 256
SPLITS = ("train", "test")


@dataclass
class VideoMeta:
    dataset: str
    video_id: str
    split: str
    num_frames: int
    height: int
    width: int
    fps: float            # EFFECTIVE fps (orig_fps / stride): use this for time-to-detection seconds
    orig_fps: float
    stride: int = 1
    has_labels: bool = False
    has_masks: bool = False


def video_dir(root, dataset: str, split: str, video_id: str) -> Path:
    return Path(root) / dataset / split / video_id


def flow_path(flow_root, dataset: str, split: str, video_id: str) -> Path:
    return Path(flow_root) / dataset / split / f"{video_id}.npy"


def save_meta(vdir: Path, meta: VideoMeta) -> None:
    (Path(vdir) / "meta.json").write_text(json.dumps(asdict(meta), indent=1))


def load_meta(vdir: Path) -> VideoMeta:
    return VideoMeta(**json.loads((Path(vdir) / "meta.json").read_text()))


def list_videos(root, dataset: str, split: str) -> list[str]:
    d = Path(root) / dataset / split
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "meta.json").exists())


def segments_from_labels(labels) -> list[tuple[int, int]]:
    """Contiguous anomalous runs as (start, end) inclusive frame indices."""
    lab = (np.asarray(labels) > 0).astype(np.int8)
    if lab.size == 0:
        return []
    d = np.diff(np.concatenate([[0], lab, [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def split_train_val(video_ids, val_fraction: float):
    """Deterministic, evenly spaced hold-out so every scene is likely represented."""
    ids = sorted(video_ids)
    n = len(ids)
    if n < 2 or val_fraction <= 0:
        return ids, []
    n_val = min(n - 1, max(1, round(val_fraction * n)))
    if n_val == 1:
        pick = {n // 2}
    else:
        pick = set(np.linspace(0, n - 1, n_val).round().astype(int).tolist())
    return [v for i, v in enumerate(ids) if i not in pick], [v for i, v in enumerate(ids) if i in pick]


def clip_frame_indices(t: int, clip_len: int, num_frames: int) -> np.ndarray:
    """Input frames for target t: [t-L, ..., t-1], clamped at 0 (replicate padding).
    Strictly causal: never touches frame t or later."""
    return np.clip(np.arange(t - clip_len, t), 0, num_frames - 1)


def train_targets(num_frames: int, clip_len: int, stride: int = 1) -> np.ndarray:
    """Training uses only fully real clips (no padding)."""
    return np.arange(clip_len, num_frames, stride)
