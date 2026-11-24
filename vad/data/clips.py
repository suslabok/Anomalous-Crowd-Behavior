"""Numpy-only video access: causal clips, flow, labels, masks for ONE video."""
from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from vad.data.common import (clip_frame_indices, flow_path, load_meta, video_dir)
from vad.data.flow import FLOW_SIZE, dequantize, farneback, quantize, small_gray

FLOW_MODES = ("none", "precomputed", "online")


class VideoStore:
    def __init__(self, processed_root, dataset, split, video_id, clip_len=8, img_size=224,
                 flow_mode="none", flow_root=None, flow_size=FLOW_SIZE, cache=64):
        if flow_mode not in FLOW_MODES:
            raise ValueError(f"flow_mode must be one of {FLOW_MODES}")
        self.dataset, self.split, self.video_id = dataset, split, video_id
        self.vdir = video_dir(processed_root, dataset, split, video_id)
        self.meta = load_meta(self.vdir)
        self.clip_len, self.img_size = clip_len, img_size
        self.flow_mode, self.flow_root, self.flow_size = flow_mode, flow_root, flow_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_n = cache
        self._flow_arr = None

    @property
    def num_frames(self) -> int:
        return self.meta.num_frames

    # ---- frames ----
    def _bgr(self, i: int) -> np.ndarray:
        hit = self._cache.get(i)
        if hit is not None:
            self._cache.move_to_end(i)
            return hit
        img = cv2.imread(str(self.vdir / "frames" / f"{i:06d}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"missing frame {i} in {self.vdir}")
        self._cache[i] = img
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return img

    def read_frame(self, i: int) -> np.ndarray:
        """float32 [3,H,W] RGB in [0,1] at img_size."""
        img = self._bgr(i)
        s = self.img_size
        if img.shape[0] != s or img.shape[1] != s:
            interp = cv2.INTER_AREA if s < img.shape[0] else cv2.INTER_LINEAR
            img = cv2.resize(img, (s, s), interpolation=interp)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0

    # ---- flow ----
    def _flow_array(self) -> np.ndarray:
        if self._flow_arr is None:
            p = flow_path(self.flow_root, self.dataset, self.split, self.video_id)
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Run: python -m vad.data.flow {self.dataset} --split {self.split} "
                    f"(or set data.flow_mode=online / none)")
            arr = np.load(p, mmap_mode="r")
            if arr.shape[-1] != self.flow_size:
                raise ValueError(f"cached flow size {arr.shape[-1]} != data.flow_size {self.flow_size}")
            self._flow_arr = arr
        return self._flow_arr

    def get_flow(self, idx: np.ndarray) -> np.ndarray:
        """Flow between consecutive INPUT frames: float32 [L-1,2,F,F]. Zero where frames repeat (padding)."""
        out = np.zeros((len(idx) - 1, 2, self.flow_size, self.flow_size), np.float32)
        for k in range(len(idx) - 1):
            a, b = int(idx[k]), int(idx[k + 1])
            # Repeated (padded) frames have no motion, so their flow stays zero.
            if b != a + 1:
                continue
            if self.flow_mode == "precomputed":
                out[k] = dequantize(self._flow_array()[a])
            else:   # online: same quantize round-trip so both modes are identical
                f = farneback(small_gray(self._bgr(a), self.flow_size), small_gray(self._bgr(b), self.flow_size))
                out[k] = dequantize(quantize(f))
        return out

    # ---- clips ----
    def get_clip(self, t: int) -> dict:
        # Input frames are all before t (causal); near the video start the first frame is repeated.
        idx = clip_frame_indices(t, self.clip_len, self.num_frames)
        d = {"clip": np.stack([self.read_frame(int(i)) for i in idx]),
             "target": self.read_frame(t),
             "t": np.int64(t)}
        if self.flow_mode != "none":
            d["flow"] = self.get_flow(idx)
        return d

    # ---- ground truth ----
    def labels(self):
        if not self.meta.has_labels:
            return None
        return np.load(self.vdir / "labels.npy")

    def masks(self):
        """uint8 [T,img,img] or None."""
        if not self.meta.has_masks:
            return None
        m = np.load(self.vdir / "masks.npz")["masks"]
        s = self.img_size
        if m.shape[-1] != s or m.shape[-2] != s:
            m = np.stack([cv2.resize(x, (s, s), interpolation=cv2.INTER_NEAREST) for x in m])
        return m
