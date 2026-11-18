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
