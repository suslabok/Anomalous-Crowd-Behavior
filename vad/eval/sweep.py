"""Operating-point sweep: how delay and false alarms trade off against threshold, smoothing, persistence.

ANALYSIS ONLY. These numbers use test labels, so never pick the reported operating point from them.
Fix the rule in advance on NORMAL validation data (e.g. percentile 99.9 + min_consecutive 3) and use
the sweep to show how sensitive the result is to that choice.
"""
from __future__ import annotations

import numpy as np

from vad.eval.report import summarize
from vad.eval.scoring import causal_smooth, choose_threshold, fit_normalizer


def save_scores(path, clip_len, val_ids, val_raws, videos) -> None:
    """Raw (un-normalised) per-frame scores, so sweeps/ablations run on CPU without re-running the model."""
    arrs = {"clip_len": np.array(clip_len), "val_ids": np.array(val_ids), "test_ids": np.array([v["id"] for v in videos]),
            "fps": np.array([v["fps"] for v in videos])}
    for i, r in zip(val_ids, val_raws):
        arrs[f"val_raw__{i}"] = np.asarray(r)
    for v in videos:
        arrs[f"raw__{v['id']}"] = np.asarray(v["raw"])
        arrs[f"labels__{v['id']}"] = np.asarray(v["labels"])
    np.savez_compressed(path, **arrs)


def load_scores(path):
    z = np.load(path)
    val_raws = [z[f"val_raw__{i}"] for i in z["val_ids"]]
    videos = [dict(id=str(i), raw=z[f"raw__{i}"], labels=z[f"labels__{i}"], fps=float(f)) for i, f in zip(z["test_ids"], z["fps"])]
    return val_raws, int(z["clip_len"]), videos


def run_sweep(val_raws, clip_len, videos, smooths=(1, 5, 9), min_consecs=(1, 3, 5),
              percentiles=(80, 90, 95, 99, 99.5, 99.9, 100), zs=(2, 3, 4, 6, 8, 12, 16), tolerance=0) -> list[dict]:
    norm = fit_normalizer([r[clip_len:] for r in val_raws])
    rows = []
    for s in smooths:
        val_s = [causal_smooth(norm(r), s)[clip_len:] for r in val_raws]
        rules = [(f"p{p}", choose_threshold(val_s, p)) for p in percentiles] + [(f"z{z}", float(z)) for z in zs]
        for m in min_consecs:
            for label, thr in rules:
                res = summarize([dict(v) for v in videos], norm, thr, s, m, [tolerance], clip_len)
                d = res["detection"][f"tol_{tolerance}"]
                rows.append(dict(smooth=s, min_consecutive=m, rule=label, threshold=float(thr),
                                 frame_auc_micro=res["frame_auc_micro_causal"],
                                 detection_rate=d["detection_rate"], delay_frames_mean=d["delay_frames_mean"],
                                 delay_frames_median=d["delay_frames_median"], delay_s_mean=d["delay_s_mean"],
                                 n_false_alarms=d["n_false_alarms"], false_alarms_per_hour=d["false_alarms_per_hour"],
                                 n_carryover=d["n_carryover"], n_censored=d["n_censored"],
                                 n_delay_segments=d["n_delay_segments"]))
    return rows


def pick_operating_points(rows, budgets=(0, 1, 3, 8)) -> list[dict]:
    """For each false-alarm budget: highest detection rate, then lowest mean delay. (Analysis only.)"""
    out = []
    for b in budgets:
        ok = [r for r in rows if r["n_false_alarms"] <= b]
        if ok:
            best = min(ok, key=lambda r: (-r["detection_rate"], np.nan_to_num(r["delay_frames_mean"], nan=1e9)))
            out.append(dict(best, budget=b))
    return out
