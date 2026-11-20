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


def test_spatial_branch_tokens_and_attention():
    from vad.models.vit import SpatialBranch
    b = SpatialBranch(img_size=64, patch=16, in_frames=4, embed_dim=32, depth=2, heads=4)
    b.set_keep_attn(True)
    t = b(torch.rand(2, 12, 64, 64))
    assert t.shape == (2, 16, 32)
    a = b.attention_maps()
    assert len(a) == 2 and a[0].shape == (2, 4, 16, 16)
    assert torch.allclose(a[0].sum(-1), torch.ones(2, 4, 16), atol=1e-4)      # each row is a distribution


def test_attention_paths_agree():
    from vad.models.vit import SpatialBranch
    b = SpatialBranch(img_size=32, patch=16, in_frames=1, embed_dim=16, depth=1, heads=2).eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        fast = b(x)
        b.set_keep_attn(True)
        slow = b(x)
    assert torch.allclose(fast, slow, atol=1e-5)


def test_spatial_predictor_shapes_and_copy_last_frame_at_init():
    from vad.models.spatial_predictor import SpatialPredictor
    m = SpatialPredictor(64, 16, 4, 4, 32, 1, 4, 2.0, residual=True).eval()
    clip = torch.rand(2, 4, 3, 64, 64)
    with torch.no_grad():
        y = m(clip)
    assert y.shape == (2, 3, 64, 64)
    assert torch.allclose(y, clip[:, -1], atol=1e-6)                           # zero-initialised residual head


def test_spatial_predictor_non_residual_range_and_single_frame():
    from vad.models.spatial_predictor import SpatialPredictor
    m = SpatialPredictor(64, 16, 4, 1, 32, 1, 4, 2.0, residual=False).eval()
    with torch.no_grad():
        y = m(torch.rand(2, 4, 3, 64, 64))
    assert y.shape == (2, 3, 64, 64) and float(y.min()) >= 0 and float(y.max()) <= 1


def test_spatial_vit_trains_and_gradients_reach_the_transformer():
    from vad.lit_module import VADLitModule
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.model.name, cfg.model.embed_dim, cfg.model.depth, cfg.model.heads = "spatial_vit", 32, 1, 4
    cfg.model.spatial_frames, cfg.data.clip_len, cfg.data.img_size = 4, 4, 64
    lit = VADLitModule(cfg)
    opt = torch.optim.SGD(lit.parameters(), lr=0.1)
    batch = {"clip": torch.rand(4, 4, 3, 64, 64), "target": torch.rand(4, 3, 64, 64)}
    for _ in range(3):
        opt.zero_grad()
        loss = lit.training_step(batch, 0)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    assert lit.model.branch.patch_embed.weight.grad.abs().sum() > 0           # zero-init head no longer blocks gradients
    assert not _has_dictconfig(dict(lit.hparams))
