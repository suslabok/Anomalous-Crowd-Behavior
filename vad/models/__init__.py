from .dual_predictor import DualBranchPredictor
from .dummy import DummyPredictor
from .spatial_predictor import SpatialPredictor
from .unet import UNetPredictor


def build_model(cfg):
    m = cfg.model
    if m.name == "dummy":
        return DummyPredictor(cfg.data.clip_len, m.hidden)
    if m.name == "unet":
        return UNetPredictor(cfg.data.clip_len, m.base_channels)
    if m.name == "spatial_vit":
        return SpatialPredictor(cfg.data.img_size, m.patch, cfg.data.clip_len, m.spatial_frames, m.embed_dim,
                                m.depth, m.heads, m.mlp_ratio, m.residual)
    if m.name == "dual_branch":
        if m.temporal_input in ("flow", "both") and cfg.data.flow_mode == "none":
            raise ValueError("model.temporal_input=%s needs optical flow: run `python -m vad.data.flow <dataset>` and "
                             "set data.flow_mode=precomputed (or online). Use model.temporal_input=diff to skip flow."
                             % m.temporal_input)
        return DualBranchPredictor(cfg.data.img_size, m.patch, cfg.data.clip_len, m.embed_dim, m.depth, m.depth_temporal,
                                   m.depth_fusion, m.heads, m.mlp_ratio, m.dual_spatial_frames, m.temporal_input,
                                   m.flow_scale, m.residual)
    raise ValueError(f"Unknown model: {m.name}")
