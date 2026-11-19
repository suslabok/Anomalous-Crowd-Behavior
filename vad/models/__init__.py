from .dummy import DummyPredictor
from .unet import UNetPredictor


def build_model(cfg):
    if cfg.model.name == "dummy":
        return DummyPredictor(cfg.data.clip_len, cfg.model.hidden)
    if cfg.model.name == "unet":
        return UNetPredictor(cfg.data.clip_len, cfg.model.base_channels)
    raise ValueError(f"Unknown model: {cfg.model.name}")
