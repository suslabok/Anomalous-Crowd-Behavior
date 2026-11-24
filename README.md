# Unsupervised Spatial-Temporal Transformer for Anomalous Crowd Behavior (COMP 493)

## Layout
- `configs/default.yaml`  config (OmegaConf, CLI-overridable)
- `vad/data/common.py`    on-disk layout, segments, train/val split, causal clip indices
- `vad/data/prepare.py`   raw -> common layout converters; `pack`/`unpack` tar on Drive
- `vad/data/flow.py`      Farneback flow, uint8 cache
- `vad/data/clips.py`     numpy core: causal clips, flow, labels, masks for one video
- `vad/data/datasets.py`  torch wrappers: TrainClips, EvalVideo, build_train_val
- `scripts/verify_dataset.py`  stats + preview image
- `tests/`                synthetic-data tests (`python -m pytest -q tests`)
- `train.py`              training entry point, auto-resumes from Drive

## Data contract
Every dataset becomes `processed/<dataset>/<split>/<video>/{meta.json, frames/*.jpg, labels.npy, masks.npz}`
at 256x256. A training sample is `clip [L,3,H,W]` (frames t-L..t-1), `target [3,H,W]` (frame t),
optional `flow [L-1,2,F,F]`. Evaluation yields one sample per frame (start replicate-padded), so scoring
is causal by construction. `meta.fps` is the EFFECTIVE fps, used to convert Δt to seconds.

## Evaluation (Phase 2)
`python scripts/evaluate.py --ckpt <run folder or .ckpt> --dataset ped2 --plots`
- Frame score = mean of top-k patch errors, z-scored with NORMAL held-out source videos, causal trailing smoothing.
- Alarm threshold = percentile of normal validation scores; fixed from the source dataset in zero-shot runs (`--source`).
- Reports frame AUC (causal micro/macro + published-style per-video-normalised), pixel AUC, time-to-detection
  (mean/median, detection rate, carry-over count, tolerance sensitivity), false alarms/hour, params/FLOPs/FPS.

### Operating-point sweep
`python scripts/sweep_operating_points.py --scores <results>/eval/<run>/scores.npz` (CPU, seconds) writes `sweep.csv`
and `tradeoff.png`. It uses test labels, so it is analysis only: fix the reported operating point on normal validation data first.

## Models
- `unet`        Phase 2 baseline (conv U-Net, early fusion over time)
- `spatial_vit` Phase 3: spatial Transformer branch + conv decoder; `model.spatial_frames=1` gives the last-frame-only ablation
Train: `python train.py data.dataset=ped2 model.name=spatial_vit train.scheduler=cosine run_name=ped2_vit_spatial`

### Warm-up / arming rule
The first `data.clip_len` frames of every video are scored from replicate-padded clips, which the model never sees in
training. Evaluation therefore treats the detector as "not armed" until it has a full clip of real history: alarms are
suppressed, and those frames are excluded from AUC, pixel AUC and the normal-footage hours. Segments that begin inside
the warm-up are counted as detected/missed but excluded from delay statistics.
- `dual_branch` Phase 4: spatial ViT (appearance) + temporal divided space-time attention (motion) + bidirectional cross-attention fusion.
  `model.temporal_input=flow|diff|both` is the with/without optical flow ablation. Needs flow for `flow`/`both`:
  `python -m vad.data.flow ped2` then `data.flow_mode=precomputed` (the flow cache lives on local disk: rebuild each Colab session).
Train: `python train.py data.dataset=ped2 model.name=dual_branch data.flow_mode=precomputed train.scheduler=cosine run_name=ped2_dual`
