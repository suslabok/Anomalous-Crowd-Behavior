"""Run in Colab (needs torch + pytorch_lightning); skipped automatically elsewhere."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")
from omegaconf import DictConfig, OmegaConf  # noqa: E402


def _has_dictconfig(x):
    if isinstance(x, DictConfig):
        return True
    return any(_has_dictconfig(v) for v in x.values()) if isinstance(x, dict) else False


def test_unet_shapes_and_range():
    from vad.models.unet import UNetPredictor
    y = UNetPredictor(4, 8)(torch.rand(2, 4, 3, 64, 64))
    assert y.shape == (2, 3, 64, 64) and float(y.min()) >= 0 and float(y.max()) <= 1


def test_gradient_loss():
    from vad.models.losses import gradient_loss
    x = torch.rand(2, 3, 16, 16)
    assert float(gradient_loss(x, x)) == 0 and float(gradient_loss(x, torch.rand_like(x))) > 0


def test_lit_module_step_and_plain_hparams():
    from vad.lit_module import VADLitModule
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.model.base_channels, cfg.data.clip_len = 8, 4
    lit = VADLitModule(cfg)
    assert not _has_dictconfig(dict(lit.hparams))                 # regression test: torch>=2.6 resume bug
    batch = {"clip": torch.rand(2, 4, 3, 64, 64), "target": torch.rand(2, 3, 64, 64)}
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_predict_video_shapes(tmp_path):
    from vad.data import prepare
    from vad.data.datasets import EvalVideo
    from vad.eval.runner import benchmark, predict_video
    from vad.models.unet import UNetPredictor
    frames = [np.full((256, 256, 3), 5 * i, np.uint8) for i in range(22)]
    lab = np.zeros(22, np.uint8); lab[10:15] = 1
    prepare.write_video(tmp_path, "toy", "test", "t", frames, 10.0, labels=lab)
    ds = EvalVideo(tmp_path, "toy", "t", clip_len=4, img_size=64)
    model = UNetPredictor(4, 8)
    pe, maps = predict_video(model, ds, torch.device("cpu"), batch_size=8, patch=16, keep_maps=True, workers=0)
    assert pe.shape == (22, 4, 4) and maps.shape == (22, 64, 64)
    b = benchmark(model, 4, 64, torch.device("cpu"), n=2)
    assert b["params"] > 0 and b["fps"] > 0
