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
    raise ValueError(f"Unknown model: {m.name}")
