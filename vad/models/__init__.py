from .dummy import DummyPredictor


def build_model(cfg):
    if cfg.model.name == "dummy":
        return DummyPredictor(cfg.data.clip_len, cfg.model.hidden)
    raise ValueError(f"Unknown model: {cfg.model.name}")
