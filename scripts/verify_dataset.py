"""Sanity-check a processed dataset and save a preview image.
    python scripts/verify_dataset.py ped2
Prints counts, fps, anomaly statistics, label/mask consistency, and writes
<results>/preview_<dataset>.png (anomalous frames with masks in red).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vad.data.common import list_videos, load_meta, segments_from_labels, video_dir
from vad.utils.paths import get_paths

# Published split sizes (warn only; subsets are legitimate, e.g. UCF-Crime).
EXPECTED = {("ped2", "train"): 16, ("ped2", "test"): 12, ("avenue", "train"): 16, ("avenue", "test"): 21,
            ("shanghaitech", "train"): 330, ("shanghaitech", "test"): 107, ("ucf_crime", "test"): 290}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--n-preview", type=int, default=6)
    a = ap.parse_args()
    p = get_paths(OmegaConf.load("configs/default.yaml"))
    ok = True
    preview = []
    for split in ("train", "test"):
        vids = list_videos(p.processed, a.dataset, split)
        if not vids:
            print(f"[{split}] none found"); continue
        metas = [load_meta(video_dir(p.processed, a.dataset, split, v)) for v in vids]
        frames = sum(m.num_frames for m in metas)
        fps = sorted({round(m.fps, 2) for m in metas})
        print(f"[{split}] videos={len(vids)} frames={frames:,} fps={fps} size={metas[0].height}x{metas[0].width}")
        exp = EXPECTED.get((a.dataset, split))
        if exp and exp != len(vids):
            print(f"  [note] published split has {exp} videos; found {len(vids)} (ok if you used a subset)")
        if split != "test":
            continue
        n_anom = n_seg = n_clean = mism = 0
        seg_len = []
        for v, m in zip(vids, metas):
            vd = video_dir(p.processed, a.dataset, split, v)
            lab = np.load(vd / "labels.npy")
            segs = segments_from_labels(lab)
            n_anom += int(lab.sum()); n_seg += len(segs); n_clean += int(not segs)
            seg_len += [e - s + 1 for s, e in segs]
            if m.has_masks:
                mk = np.load(vd / "masks.npz")["masks"]
                mism += int((mk.reshape(len(mk), -1).any(1) != (lab > 0)).sum())
                if segs and len(preview) < a.n_preview:
                    t = (segs[0][0] + segs[0][1]) // 2
                    img = cv2.imread(str(vd / "frames" / f"{t:06d}.jpg"))
                    img[mk[t] > 0] = (0.5 * img[mk[t] > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
                    preview.append(img)
        print(f"  anomalous frames={n_anom:,} ({100 * n_anom / frames:.1f}%), segments={n_seg}, "
              f"videos without anomalies={n_clean}, mean segment length={np.mean(seg_len) if seg_len else 0:.1f} frames")
        if any(m.has_masks for m in metas):
            print(f"  label/mask mismatches: {mism}")
            ok &= mism == 0
    if preview:
        out = p.results / f"preview_{a.dataset}.png"
        cv2.imwrite(str(out), np.hstack(preview)); print(f"preview -> {out}")
    print("RESULT:", "OK" if ok else "CHECK WARNINGS ABOVE")


if __name__ == "__main__":
    main()
