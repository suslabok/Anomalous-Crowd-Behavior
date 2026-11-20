"""Delay / false-alarm trade-off from saved scores (CPU only, takes seconds).

    python scripts/sweep_operating_points.py --scores <results>/eval/<run>/scores.npz

ANALYSIS ONLY: it uses test labels. Choose the reported operating point from NORMAL validation data in advance.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vad.eval.plots import plot_tradeoff
from vad.eval.sweep import load_scores, pick_operating_points, run_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tolerance", type=int, default=0)
    a = ap.parse_args()
    val_raws, clip_len, videos = load_scores(a.scores)
    rows = run_sweep(val_raws, clip_len, videos, tolerance=a.tolerance)
    out = Path(a.scores).parent
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot_tradeoff(rows, out / "tradeoff.png")
    print(f"{len(rows)} operating points -> {out / 'sweep.csv'}, {out / 'tradeoff.png'}\n")
    print("Best point per false-alarm budget (analysis only, NOT a reportable operating point):")
    print(f"{'budget':>6} {'rule':>6} {'smooth':>6} {'persist':>7} {'FA':>3} {'detected':>8} {'mean delay':>10} {'median':>7}")
    for r in pick_operating_points(rows):
        print(f"{r['budget']:>6} {r['rule']:>6} {r['smooth']:>6} {r['min_consecutive']:>7} {r['n_false_alarms']:>3} "
              f"{r['detection_rate']:>8.2f} {r['delay_frames_mean']:>10.1f} {r['delay_frames_median']:>7.1f}")


if __name__ == "__main__":
    main()
