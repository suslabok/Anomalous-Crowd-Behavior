"""Evaluate a checkpoint under the proposal's ONLINE protocol.

    python scripts/evaluate.py --ckpt <file or run folder> --dataset ped2 [--source ped2] [eval.smooth=5 ...]

  --source   dataset whose held-out NORMAL videos fit the score normalizer and the alarm threshold
             (defaults to the dataset the model was trained on).
  --dataset  dataset being tested. source != dataset  ->  zero-shot transfer (Protocol B):
             normalizer and threshold stay fixed from the source and are never re-tuned on the target.
Writes metrics.json, per_video.csv and curves.png to <results>/eval/<name>/.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vad.data.common import list_videos, split_train_val
from vad.data.datasets import EvalVideo
from vad.eval.metrics import PixelAcc, blur_maps
from vad.eval.report import clean, summarize
from vad.eval.runner import benchmark, predict_video
from vad.eval.sweep import save_scores
from vad.eval.scoring import causal_smooth, choose_threshold, fit_normalizer, topk_mean
from vad.lit_module import VADLitModule
from vad.utils.paths import get_paths


def resolve_ckpt(p) -> Path:
    p = Path(p)
    if p.is_dir():
        best = sorted(f for f in p.glob("*.ckpt") if f.name != "last.ckpt")
        p = best[0] if best else p / "last.ckpt"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["hyper_parameters"]["cfg"])
    lit = VADLitModule(cfg)
    lit.load_state_dict(ck["state_dict"])
    return lit.model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    args, overrides = ap.parse_known_args()

    base = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    ev = base.eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_ckpt(args.ckpt)
    model, cfg = load_model(ckpt, device)
    cfg.paths = base.paths
    paths = get_paths(cfg)
    dc = cfg.data
    source = args.source or dc.dataset
    target = args.dataset or dc.dataset
    name = args.name or f"{ckpt.parent.name}_{source}_to_{target}"
    out_dir = paths.results / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] ckpt={ckpt}\n[eval] source={source} target={target} device={device}")

    def make_ds(dataset, vid, split):
        return EvalVideo(paths.processed, dataset, vid, dc.clip_len, dc.img_size, dc.flow_mode, paths.flow,
                         dc.flow_size, split=split)

    # 1) normalizer + threshold from held-out NORMAL source videos (warm-up frames with padded clips are dropped)
    _, val_ids = split_train_val(list_videos(paths.processed, source, "train"), dc.val_fraction)
    if not val_ids:
        raise RuntimeError(f"no held-out normal videos for source '{source}'")
    raws = []
    for v in val_ids:
        pe, _ = predict_video(model, make_ds(source, v, "train"), device, args.batch_size, ev.patch, False)
        raws.append(topk_mean(pe, ev.topk))
    norm = fit_normalizer([r[dc.clip_len:] for r in raws])
    thr = choose_threshold([causal_smooth(norm(r), ev.smooth)[dc.clip_len:] for r in raws], ev.thr_percentile)
    norm.save(out_dir / "normalizer.json")
    val_raws = raws
    print(f"[eval] fitted on {len(val_ids)} normal videos {val_ids}: mean={norm.mean:.5f} std={norm.std:.5f} thr={thr:.3f}")

    # 2) score the target test videos
    videos, pix, pix_anom = [], PixelAcc(ev.pixel_samples), PixelAcc(ev.pixel_samples)
    for v in list_videos(paths.processed, target, "test"):
        ds = make_ds(target, v, "test")
        if ds.labels is None:
            raise RuntimeError(f"{target}/{v} has no frame labels")
        masks = ds.store.masks()
        pe, maps = predict_video(model, ds, device, args.batch_size, ev.patch, masks is not None)
        videos.append(dict(id=v, raw=topk_mean(pe, ev.topk), labels=ds.labels, fps=ds.meta.fps))
        if masks is not None:
            blurred = blur_maps(maps, ev.map_sigma)
            pix.add(blurred, masks)
            anom = ds.labels > 0
            if anom.any():
                pix_anom.add(blurred[anom], masks[anom])
        print(f"  scored {v} ({len(ds)} frames)")

    save_scores(out_dir / "scores.npz", dc.clip_len, val_ids, val_raws, videos)
    m = summarize(videos, norm, thr, ev.smooth, ev.min_consecutive, list(ev.tolerances), dc.clip_len)
    m["pixel_auc"] = pix.auc()                       # all test frames pooled (background pixels make this easy)
    m["pixel_auc_anomalous_frames"] = pix_anom.auc()  # only frames that contain an anomaly (harder, more informative)
    m["efficiency"] = benchmark(model, dc.clip_len, dc.img_size, device)
    m.update(source=source, target=target, ckpt=str(ckpt), eval_cfg=OmegaConf.to_container(ev),
             normalizer=dict(mean=norm.mean, std=norm.std))
    (out_dir / "metrics.json").write_text(json.dumps(clean(m), indent=1))
    with open(out_dir / "per_video.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frames", "auc", "segments", "segment_starts", "detected", "delay_frames", "censored", "false_alarms"])
        for p in m["per_video"]:
            w.writerow([p["id"], p["n_frames"], p["auc"], p["n_segments"], p["segment_starts"], p["n_detected"], p["delay_frames"], p["censored"], p["n_false_alarms"]])
    if args.plots:
        from vad.eval.plots import plot_score_curves
        plot_score_curves(videos, thr, out_dir / "curves.png")

    d0 = m["detection"]["tol_0"]
    print("\n==== RESULTS", name, "====")
    print(f"frame AUC  micro(causal)={m['frame_auc_micro_causal']:.4f}  macro(causal)={m['frame_auc_macro_causal']:.4f}  "
          f"micro(per-video norm, published-style)={m['frame_auc_micro_pervideo_norm']:.4f}")
    print(f"pixel AUC  all frames={m['pixel_auc']:.4f}  anomalous frames only={m['pixel_auc_anomalous_frames']:.4f}")
    print(f"detection  {d0['n_detected']}/{d0['n_segments']} segments detected "
          f"({d0['n_censored']} start inside the first {dc.clip_len} frames: delay undefined, excluded from delay stats)")
    print(f"delay      over {d0['n_delay_segments']} genuine onsets: mean {d0['delay_frames_mean']:.1f} frames "
          f"({d0['delay_s_mean']:.2f} s), median {d0['delay_frames_median']:.1f} frames, carry-over={d0['n_carryover']}")
    print(f"false alarms {d0['n_false_alarms']} on {d0['normal_hours'] * 3600:.0f} s of normal footage "
          f"({d0['false_alarms_per_hour']:.1f}/hour)")
    e = m["efficiency"]
    print(f"model      {e['params'] / 1e6:.2f} M params, {e['fps']:.0f} FPS, {e['latency_ms']:.1f} ms/prediction ({e['device']})")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
