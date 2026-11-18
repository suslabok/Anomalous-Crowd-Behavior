"""Optical flow: Farneback, uint8 quantization, per-video precompute (CLI)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vad.data.common import flow_path, list_videos, video_dir

FLOW_SIZE = 128
FLOW_MAX = 12.0     # pixels at FLOW_SIZE resolution; larger motion is clipped


def small_gray(img_bgr: np.ndarray, size: int = FLOW_SIZE) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def farneback(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    f = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return f.transpose(2, 0, 1).astype(np.float32)          # [2,H,W]


def quantize(flow: np.ndarray) -> np.ndarray:
    q = np.round(np.clip(flow, -FLOW_MAX, FLOW_MAX) / FLOW_MAX * 127.0) + 128.0
    return q.astype(np.uint8)


def dequantize(q: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) - 128.0) / 127.0 * FLOW_MAX


def precompute_video(vdir, out_path, size: int = FLOW_SIZE) -> int:
    files = sorted((Path(vdir) / "frames").glob("*.jpg"))
    prev, out = None, []
    for f in files:
        g = small_gray(cv2.imread(str(f)), size)
        if prev is not None:
            out.append(quantize(farneback(prev, g)))
        prev = g
    arr = np.stack(out) if out else np.zeros((0, 2, size, size), np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return len(out)


def precompute_dataset(processed_root, flow_root, dataset, split, size=FLOW_SIZE) -> None:
    from tqdm import tqdm
    for vid in tqdm(list_videos(processed_root, dataset, split), desc=f"flow {dataset}/{split}"):
        out = flow_path(flow_root, dataset, split, vid)
        if out.exists():
            continue
        precompute_video(video_dir(processed_root, dataset, split, vid), out, size)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from vad.utils.paths import get_paths

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--size", type=int, default=FLOW_SIZE)
    a = ap.parse_args()
    p = get_paths(OmegaConf.load("configs/default.yaml"))
    for s in (["train", "test"] if a.split == "both" else [a.split]):
        precompute_dataset(p.processed, p.flow, a.dataset, s, a.size)
