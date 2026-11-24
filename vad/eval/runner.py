"""Torch side of evaluation: run a model over a video, benchmark speed."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_video(model, ds, device, batch_size: int = 32, patch: int = 16, keep_maps: bool = False, workers: int = 2):
    """Predict every frame of an EvalVideo from its past only. Returns (patch_err [T,h,w], maps [T,H,W] | None)."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    use_amp = device.type == "cuda"
    pes, maps = [], []
    for b in dl:
        clip = b["clip"].to(device, non_blocking=True)
        target = b["target"].to(device, non_blocking=True)
        flow = b["flow"].to(device, non_blocking=True) if "flow" in b else None
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(clip, flow)
        err = (pred.float() - target.float()).pow(2).mean(dim=1)             # [B,H,W]
        pes.append(F.avg_pool2d(err.unsqueeze(1), patch).squeeze(1).cpu().numpy())
        if keep_maps:
            maps.append(err.cpu().numpy().astype(np.float16))
    return np.concatenate(pes), (np.concatenate(maps) if keep_maps else None)


@torch.no_grad()
def benchmark(model, clip_len: int, img_size: int, device, n: int = 30, flow_size: int = 128) -> dict:
    """Model-only speed (batch of 1 frame-prediction), params and FLOPs per prediction. Excludes the cost of
    computing optical flow (report that separately); a zero flow tensor is fed to flow-based models."""
    model.eval()
    x = torch.rand(1, clip_len, 3, img_size, img_size, device=device)
    flow = torch.zeros(1, clip_len - 1, 2, flow_size, flow_size, device=device)
    use_amp = device.type == "cuda"

    def run():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            return model(x, flow)

    for _ in range(5):
        run()
    if use_amp:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        run()
    if use_amp:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    out = dict(params=int(sum(p.numel() for p in model.parameters())), latency_ms=dt * 1000, fps=1.0 / dt,
               device=str(device), flops=None)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            model(x.float(), flow)
        out["flops"] = int(fc.get_total_flops())
    except Exception:
        pass
    return out
