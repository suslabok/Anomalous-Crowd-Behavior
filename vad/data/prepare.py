"""Convert raw datasets into the common layout (see vad/data/common.py).

    python -m vad.data.prepare prepare ped2 --raw /content/data/raw
    python -m vad.data.prepare pack    ped2        # processed -> Drive tar (do once)
    python -m vad.data.prepare unpack  ped2        # Drive tar -> local disk (every session)

Raw layouts expected (searched recursively under --raw):
  ped2          UCSDped2/{Train/TrainXXX/*.tif, Test/TestXXX/*.tif, Test/TestXXX_gt/*.bmp}
  avenue        training_videos/*.avi, testing_videos/*.avi, and *_label.mat (volLabel) anywhere
  shanghaitech  training/{frames/<vid>/*.jpg | videos/*.avi}, testing/{frames/<vid>, test_frame_mask/<vid>.npy,
                test_pixel_mask/<vid>.npy}
  ucf_crime     *.mp4 test videos + Temporal_Anomaly_Annotation*.txt   (evaluation only)
"""
from __future__ import annotations

import argparse
import re
import tarfile
from pathlib import Path

import cv2
import numpy as np

from vad.data.common import STORE_SIZE, VideoMeta, save_meta, video_dir

IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


# ------------------------------------------------------------------ helpers
def _natural(p) -> list:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", Path(p).name)]


def _find_dir(raw, name: str) -> Path:
    raw = Path(raw)
    if (raw / name).is_dir():
        return raw / name
    for p in sorted(raw.rglob(name)):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"directory '{name}' not found under {raw}. Contents: "
                            f"{[x.name for x in raw.iterdir()][:20] if raw.exists() else 'raw dir missing'}")


def _to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _fit(arr, n: int, name: str, vid: str, pad_value=0):
    arr = list(arr)
    if len(arr) == n:
        return arr
    print(f"  [warn] {vid}: {name} length {len(arr)} != frames {n}; "
          f"{'truncating' if len(arr) > n else 'padding'}")
    if len(arr) > n:
        return arr[:n]
    fill = np.zeros_like(arr[0]) if arr else pad_value
    return arr + [fill] * (n - len(arr))


def _open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, (fps if fps and fps > 0 else 25.0)


def _video_frames(cap, stride=1):
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield f
        i += 1
    cap.release()


def write_video(root, dataset, split, vid, frames, orig_fps, stride=1, labels=None, masks=None,
                size=STORE_SIZE) -> int:
    vdir = video_dir(root, dataset, split, vid)
    fdir = vdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in frames:
        img = cv2.resize(_to_bgr(img), (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(fdir / f"{n:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    if labels is not None:
        lab = np.asarray(_fit(np.asarray(labels).astype(np.uint8), n, "labels", vid), np.uint8)
        np.save(vdir / "labels.npy", lab)
    if masks is not None:
        ms = [cv2.resize((np.asarray(m) > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
              for m in masks]
        np.savez_compressed(vdir / "masks.npz", masks=np.stack(_fit(ms, n, "masks", vid)))
    save_meta(vdir, VideoMeta(dataset, vid, split, n, size, size, orig_fps / stride, orig_fps, stride,
                              labels is not None, masks is not None))
    return n


# ------------------------------------------------------------------ datasets
def prepare_ped2(raw, out, size=STORE_SIZE, fps=10.0):
    root = _find_dir(raw, "UCSDped2")
    for split, sub in (("train", "Train"), ("test", "Test")):
        vids = sorted((p for p in (root / sub).iterdir() if p.is_dir() and not p.name.endswith("_gt")), key=_natural)
        for v in vids:
            files = sorted((f for f in v.iterdir() if f.suffix.lower() in IMG_EXT), key=_natural)
            frames = (cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files)
            labels = masks = None
            if split == "test":
                gt = root / sub / f"{v.name}_gt"
                if not gt.is_dir():
                    raise FileNotFoundError(f"missing ground truth folder {gt}")
                mf = sorted((f for f in gt.iterdir() if f.suffix.lower() in IMG_EXT), key=_natural)
                masks = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in mf]
                labels = [int((m > 0).any()) for m in masks]
            n = write_video(out, "ped2", split, v.name, frames, fps, 1, labels, masks, size)
            print(f"ped2 {split} {v.name}: {n} frames")


def _avenue_masks(mat_path, n_frames):
    from scipy.io import loadmat
    vol = loadmat(str(mat_path))["volLabel"]
    if vol.dtype == object:
        return [np.asarray(vol[0, i]) for i in range(vol.shape[1])]
    if vol.ndim == 3:
        axes = [a for a in range(3) if vol.shape[a] == n_frames]
        if axes:
            return list(np.moveaxis(vol, axes[0], 0))
    raise ValueError(f"unexpected volLabel format in {mat_path}: dtype={vol.dtype} shape={vol.shape}")


def prepare_avenue(raw, out, size=STORE_SIZE):
    mats = {int(re.match(r"(\d+)", p.stem).group(1)): p for p in Path(raw).rglob("*_label.mat")}
    for split, d in (("train", "training_videos"), ("test", "testing_videos")):
        for vp in sorted(_find_dir(raw, d).glob("*.avi"), key=_natural):
            cap, fps = _open_video(vp)
            labels = masks = None
            if split == "test":
                k = int(re.match(r"(\d+)", vp.stem).group(1))
                if k not in mats:
                    raise FileNotFoundError(f"no *_label.mat for test video {vp.name}")
                masks = _avenue_masks(mats[k], int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                labels = [int((np.asarray(m) > 0).any()) for m in masks]
            n = write_video(out, "avenue", split, vp.stem, _video_frames(cap), fps, 1, labels, masks, size)
            print(f"avenue {split} {vp.stem}: {n} frames @ {fps:.1f} fps")


def prepare_shanghaitech(raw, out, size=STORE_SIZE, default_fps=24.0):
    tr, te = _find_dir(raw, "training"), _find_dir(raw, "testing")
    avis = sorted(tr.rglob("*.avi"))
    fps = default_fps
    if avis:
        cap, fps = _open_video(avis[0]); cap.release()
    else:
        print(f"[warn] no .avi found to read fps from; assuming {default_fps} fps. "
              f"VERIFY this: it scales every time-to-detection value in seconds.")
    # train
    if (tr / "frames").is_dir():
        for v in sorted((p for p in (tr / "frames").iterdir() if p.is_dir()), key=_natural):
            files = sorted((f for f in v.iterdir() if f.suffix.lower() in IMG_EXT), key=_natural)
            n = write_video(out, "shanghaitech", "train", v.name, (cv2.imread(str(f)) for f in files), fps, 1, None, None, size)
            print(f"shanghaitech train {v.name}: {n}")
    else:
        for vp in sorted((tr / "videos").glob("*.avi"), key=_natural):
            cap, f_ = _open_video(vp)
            n = write_video(out, "shanghaitech", "train", vp.stem, _video_frames(cap), f_, 1, None, None, size)
            print(f"shanghaitech train {vp.stem}: {n}")
    # test
    for v in sorted((p for p in (te / "frames").iterdir() if p.is_dir()), key=_natural):
        files = sorted((f for f in v.iterdir() if f.suffix.lower() in IMG_EXT), key=_natural)
        lab = np.load(te / "test_frame_mask" / f"{v.name}.npy")
        pm = te / "test_pixel_mask" / f"{v.name}.npy"
        masks = list(np.load(pm)) if pm.exists() else None
        n = write_video(out, "shanghaitech", "test", v.name, (cv2.imread(str(f)) for f in files), fps, 1, lab, masks, size)
        print(f"shanghaitech test {v.name}: {n}")


def _subsample(items, k):
    if k is None or k >= len(items):
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, k).round().astype(int)]


def prepare_ucf_crime(raw, out, size=STORE_SIZE, stride=1, annotations=None,
                      max_anomalous=None, max_normal=None):
    ann = Path(annotations) if annotations else next(Path(raw).rglob("*Temporal_Anomaly_Annotation*.txt"), None)
    if ann is None or not Path(ann).exists():
        raise FileNotFoundError("temporal annotation txt not found; pass --annotations")
    vids = {p.stem: p for p in Path(raw).rglob("*.mp4")}
    segs, normals = {}, []
    for line in Path(ann).read_text().splitlines():
        tok = line.split()
        if len(tok) < 2:
            continue
        name, cls = Path(tok[0]).stem, tok[1]
        nums = [int(x) for x in tok[2:]]
        s = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2) if nums[i] >= 0 and nums[i + 1] >= 0]
        if cls.lower() == "normal" or not s:
            normals.append(name)
        else:
            segs[name] = s
    for stem, p in vids.items():   # normal test videos that are not in the annotation file
        if "Testing_Normal" in str(p.parent) and stem not in segs and stem not in normals:
            normals.append(stem)
    anom = _subsample(sorted(segs), max_anomalous)
    norm = _subsample(sorted(set(normals)), max_normal)
    for name in anom + norm:
        if name not in vids:
            print(f"  [warn] video {name} not found, skipping")
            continue
        cap, fps = _open_video(vids[name])
        frames = list(_video_frames(cap, 1))
        lab = np.zeros(len(frames), np.uint8)
        for s, e in segs.get(name, []):
            lab[s:e + 1] = 1
        n = write_video(out, "ucf_crime", "test", name, frames[::stride], fps, stride, lab[::stride], None, size)
        print(f"ucf_crime test {name}: {n} frames, anomalous={int(lab[::stride].sum())}")


# ------------------------------------------------------------------ pack / unpack
def pack(processed_root, archive_dir, dataset):
    archive_dir = Path(archive_dir); archive_dir.mkdir(parents=True, exist_ok=True)
    dst = archive_dir / f"{dataset}_processed.tar"
    with tarfile.open(dst, "w") as tf:
        tf.add(Path(processed_root) / dataset, arcname=dataset)
    print(f"packed -> {dst} ({dst.stat().st_size / 1e9:.2f} GB)")


def unpack(processed_root, archive_dir, dataset):
    src = Path(archive_dir) / f"{dataset}_processed.tar"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run `prepare` then `pack` once.")
    Path(processed_root).mkdir(parents=True, exist_ok=True)
    with tarfile.open(src) as tf:
        tf.extractall(processed_root)
    print(f"unpacked {src} -> {processed_root}")


def main():
    from omegaconf import OmegaConf
    from vad.utils.paths import get_paths

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "pack", "unpack"])
    ap.add_argument("dataset", choices=["ped2", "avenue", "shanghaitech", "ucf_crime"])
    ap.add_argument("--raw", default=None)
    ap.add_argument("--size", type=int, default=STORE_SIZE)
    ap.add_argument("--stride", type=int, default=1, help="ucf_crime temporal subsampling")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--max-anomalous", type=int, default=None)
    ap.add_argument("--max-normal", type=int, default=None)
    a = ap.parse_args()
    p = get_paths(OmegaConf.load("configs/default.yaml"))
    if a.cmd == "pack":
        return pack(p.processed, p.archives, a.dataset)
    if a.cmd == "unpack":
        return unpack(p.processed, p.archives, a.dataset)
    raw = a.raw or p.raw_data
    if a.dataset == "ped2":
        prepare_ped2(raw, p.processed, a.size)
    elif a.dataset == "avenue":
        prepare_avenue(raw, p.processed, a.size)
    elif a.dataset == "shanghaitech":
        prepare_shanghaitech(raw, p.processed, a.size)
    else:
        prepare_ucf_crime(raw, p.processed, a.size, a.stride, a.annotations, a.max_anomalous, a.max_normal)


if __name__ == "__main__":
    main()
