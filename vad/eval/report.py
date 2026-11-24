"""Turn raw per-frame scores into the full metric report (numpy only)."""
from __future__ import annotations

import math

import numpy as np

from vad.eval.metrics import (frame_auc, false_alarms, minmax, summarize_detection, time_to_detection, _auc)
from vad.eval.scoring import Normalizer, alarm_signal, causal_smooth


def clean(o):
    """Make nested results JSON-safe (numpy -> python, NaN -> None)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) or math.isinf(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def summarize(videos, norm: Normalizer, thr: float, smooth: int, min_consecutive: int, tolerances, warmup: int = 0) -> dict:
    """videos: list of dict(id, raw[T], labels[T], fps). Adds 'score' (normalized + causally smoothed) and 'alarm'
    to each dict in place and returns the metrics dict.

    warmup: the detector is not armed until it has seen `warmup` real frames (a full clip). Alarms are suppressed
    before that, and those frames are left out of AUC and of the normal-footage hours. Without this, the
    replicate-padded clips at the start of every video (which the model never saw in training) cause spurious alarms."""
    w = warmup
    for v in videos:
        v["score"] = causal_smooth(norm(v["raw"]), smooth)
        alarm = alarm_signal(v["score"], thr, min_consecutive)
        alarm[:w] = False
        v["alarm"] = alarm
    full = [v["labels"] for v in videos]
    labels = [l[w:] for l in full]
    scores = [v["score"][w:] for v in videos]
    pervideo = [minmax(causal_smooth(v["raw"], smooth)[w:]) for v in videos]   # published-style, NOT causal
    m = dict(
        threshold=thr, n_videos=len(videos), n_frames=int(sum(len(l) for l in labels)), warmup=w,
        frame_auc_micro_causal=frame_auc(scores, labels, "micro"),
        frame_auc_macro_causal=frame_auc(scores, labels, "macro"),
        frame_auc_micro_pervideo_norm=frame_auc(pervideo, labels, "micro"),
        detection={f"tol_{t}": summarize_detection([v["alarm"] for v in videos], full, [v["fps"] for v in videos], t, w)
                   for t in tolerances},
    )
    per = []
    for v in videos:
        recs = time_to_detection(v["alarm"], v["labels"], v["fps"], 0, w)
        fa, hrs = false_alarms(v["alarm"], v["labels"], v["fps"], 0, w)
        per.append(dict(id=v["id"], n_frames=len(v["labels"]), auc=_auc(v["labels"][w:], v["score"][w:]),
                        n_segments=len(recs), n_detected=sum(r["detected"] for r in recs),
                        segment_starts=[r["start"] for r in recs], censored=[r["censored"] for r in recs],
                        delay_frames=[r["delay_frames"] for r in recs], n_false_alarms=fa))
    m["per_video"] = per
    return m
