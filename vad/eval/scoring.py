"""Causal anomaly scoring (numpy only). Nothing here looks at frames after t."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def patch_pool(err: np.ndarray, patch: int) -> np.ndarray:
    """[T,H,W] pixel errors -> [T,H/p,W/p] patch errors (mean over each patch)."""
    t, h, w = err.shape
    assert h % patch == 0 and w % patch == 0, "H and W must be divisible by the patch size"
    return err.reshape(t, h // patch, patch, w // patch, patch).mean(axis=(2, 4))


def topk_mean(patch_err: np.ndarray, k: int) -> np.ndarray:
    """[T,h,w] -> [T]: mean of the k largest patch errors of each frame."""
    flat = patch_err.reshape(len(patch_err), -1)
    k = max(1, min(k, flat.shape[1]))
    return np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:].mean(axis=1)


@dataclass
class Normalizer:
    """z-score with statistics from NORMAL source-dataset video (never from the test video)."""
    mean: float
    std: float

    def __call__(self, raw):
        return (np.asarray(raw, dtype=np.float64) - self.mean) / max(self.std, 1e-8)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(path) -> "Normalizer":
        return Normalizer(**json.loads(Path(path).read_text()))


def fit_normalizer(raw_scores_list) -> Normalizer:
    x = np.concatenate([np.asarray(r, dtype=np.float64) for r in raw_scores_list])
    return Normalizer(float(x.mean()), float(max(x.std(), 1e-8)))


def causal_smooth(x, window: int) -> np.ndarray:
    """Trailing moving average: out[t] = mean(x[max(0,t-window+1) .. t])."""
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) == 0:
        return x.copy()
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - window, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def choose_threshold(normal_score_arrays, percentile: float) -> float:
    return float(np.percentile(np.concatenate([np.asarray(a) for a in normal_score_arrays]), percentile))


def alarm_signal(scores, thr: float, min_consecutive: int = 1) -> np.ndarray:
    """alarm[t] is True when the score has exceeded thr for min_consecutive frames ending at t."""
    above = np.asarray(scores) > thr
    if min_consecutive <= 1:
        return above
    out = np.zeros_like(above)
    run = 0
    for t, a in enumerate(above):
        run = run + 1 if a else 0
        out[t] = run >= min_consecutive
    return out
