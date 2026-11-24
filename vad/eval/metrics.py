"""AUC, time-to-detection and false-alarm metrics (numpy/sklearn)."""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from vad.data.common import segments_from_labels


# ------------------------------------------------------------------ AUC
def _auc(y, s) -> float:
    y = np.asarray(y) > 0
    if y.all() or not y.any():
        return float("nan")
    return float(roc_auc_score(y, s))


def frame_auc(scores_list, labels_list, mode: str = "micro") -> float:
    """micro: all frames pooled. macro: mean of per-video AUC over videos containing both classes."""
    if mode == "micro":
        return _auc(np.concatenate(labels_list), np.concatenate(scores_list))
    per = [_auc(l, s) for s, l in zip(scores_list, labels_list)]
    per = [p for p in per if not np.isnan(p)]
    return float(np.mean(per)) if per else float("nan")


def minmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)


def blur_maps(maps: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return maps.astype(np.float32)
    return np.stack([cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma) for m in maps])


class PixelAcc:
    """Pooled pixel-level AUC with uniform per-video subsampling (keeps memory bounded)."""

    def __init__(self, per_video: int = 300_000, seed: int = 0):
        self.per_video, self.rng, self.s, self.y = per_video, np.random.default_rng(seed), [], []

    def add(self, maps, masks) -> None:
        m = np.asarray(maps, dtype=np.float32).reshape(-1)
        y = np.asarray(masks).reshape(-1) > 0
        if m.size > self.per_video:
            keep = self.rng.random(m.size) < self.per_video / m.size
            m, y = m[keep], y[keep]
        self.s.append(m); self.y.append(y)

    def auc(self) -> float:
        return _auc(np.concatenate(self.y), np.concatenate(self.s)) if self.s else float("nan")


# ------------------------------------------------------------------ time to detection
def time_to_detection(alarm, labels, fps: float, tolerance: int = 0, warmup: int = 0) -> list[dict]:
    """One record per annotated anomalous segment [s,e].

    Detection = first frame t with alarm[t] in [s - tolerance, e]. Delay = t - s (frames), signed only
    when tolerance > 0. With tolerance 0 an alarm before the onset is NOT a negative delay (it is a false
    alarm, see false_alarms()). `carryover` marks detections where the alarm was already on just before
    the window, i.e. a pre-onset alarm that persisted into the segment (delay is then optimistic).
    `censored` marks segments that start inside the first `warmup` frames: the anomaly is already in progress
    when the video starts (or before the model has any history), so there is no onset to be late for. Such
    segments count toward the detection rate but are excluded from delay statistics."""
    alarm = np.asarray(alarm, dtype=bool)
    out = []
    for s, e in segments_from_labels(labels):
        # Window in which an alarm counts as detecting this segment (it starts `tolerance` frames early).
        lo = max(0, s - tolerance)
        hits = np.flatnonzero(alarm[lo:e + 1])
        if len(hits) == 0:
            out.append(dict(start=s, end=e, detected=False, delay_frames=None, delay_s=None, carryover=False,
                            censored=bool(s < warmup)))
            continue
        t = lo + int(hits[0])
        cens = bool(s < warmup)
        out.append(dict(start=s, end=e, detected=True, delay_frames=t - s, delay_s=(t - s) / fps,
                        carryover=bool(t == lo and lo > 0 and alarm[lo - 1]) and not cens, censored=cens))
    return out


def false_alarms(alarm, labels, fps: float, tolerance: int = 0, warmup: int = 0) -> tuple[int, float]:
    """(#alarm events on normal footage, hours of normal footage). An event is a rising edge of the alarm
    signal. Frames within `tolerance` before an annotated onset are not counted as normal, and neither are the
    first `warmup` frames (the detector is not armed yet: its clips there are replicate-padded)."""
    alarm = np.asarray(alarm, dtype=bool)
    labels = np.asarray(labels) > 0
    normal = ~labels
    for s, _ in segments_from_labels(labels):
        normal[max(0, s - tolerance):s] = False
    normal[:warmup] = False
    # One alarm event = the moment the alarm switches on, so a long alarm counts once.
    edges = alarm & ~np.concatenate([[False], alarm[:-1]])
    return int((edges & normal).sum()), float(normal.sum() / fps / 3600.0)


def summarize_detection(alarms, labels_list, fps_list, tolerance: int = 0, warmup: int = 0) -> dict:
    recs, n_fa, hours = [], 0, 0.0
    for a, l, f in zip(alarms, labels_list, fps_list):
        recs += time_to_detection(a, l, f, tolerance, warmup)
        fa, h = false_alarms(a, l, f, tolerance, warmup)
        n_fa += fa; hours += h
    det = [r for r in recs if r["detected"]]
    dly = [r for r in det if not r["censored"]]            # genuine onsets only
    fr = [r["delay_frames"] for r in dly]
    sec = [r["delay_s"] for r in dly]
    nan = float("nan")
    return dict(n_segments=len(recs), n_detected=len(det),
                detection_rate=len(det) / len(recs) if recs else nan,
                delay_frames_mean=float(np.mean(fr)) if fr else nan, delay_frames_median=float(np.median(fr)) if fr else nan,
                delay_s_mean=float(np.mean(sec)) if sec else nan, delay_s_median=float(np.median(sec)) if sec else nan,
                n_censored=sum(r["censored"] for r in recs), n_delay_segments=len(dly),
                n_carryover=sum(r["carryover"] for r in dly),
                n_false_alarms=n_fa, normal_hours=hours,
                false_alarms_per_hour=n_fa / hours if hours > 0 else nan)
