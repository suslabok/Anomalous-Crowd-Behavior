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


# ---------------------------------------------------------------- Phase 4: temporal branch, fusion, dual model
def test_cross_attention_shapes_and_rows_sum_to_one():
    from vad.models.vit import CrossAttention
    ca = CrossAttention(32, 4)
    ca.keep_attn = True
    y = ca(torch.rand(2, 5, 32), torch.rand(2, 7, 32))
    assert y.shape == (2, 5, 32) and ca.last_attn.shape == (2, 4, 5, 7)
    assert torch.allclose(ca.last_attn.sum(-1), torch.ones(2, 4, 5), atol=1e-4)


def test_cross_attention_paths_agree():
    from vad.models.vit import CrossAttention
    ca = CrossAttention(16, 2).eval()
    x, c = torch.rand(2, 4, 16), torch.rand(2, 6, 16)
    with torch.no_grad():
        fast = ca(x, c)
        ca.keep_attn = True
        slow = ca(x, c)
    assert torch.allclose(fast, slow, atol=1e-5)


def test_temporal_branch_shape_and_time_order_sensitivity():
    from vad.models.temporal import TemporalBranch
    t = TemporalBranch(img_size=64, patch=16, in_ch=2, steps=3, embed_dim=32, depth=1, heads=4).eval()
    x = torch.rand(2, 3, 2, 64, 64)
    with torch.no_grad():
        a = t(x)
        assert a.shape == (2, 16, 32)
        t.tpos.data.normal_(0, 1.0)                       # make the time embedding clearly non-trivial
        assert not torch.allclose(t(x), t(x.flip(1)), atol=1e-5)    # reversing time changes the motion tokens


def _dual(temporal_input="flow", residual=True):
    from vad.models.dual_predictor import DualBranchPredictor
    return DualBranchPredictor(img_size=64, patch=16, clip_len=4, embed_dim=32, depth_spatial=1, depth_temporal=1,
                               depth_fusion=1, heads=4, mlp_ratio=2.0, spatial_frames=1,
                               temporal_input=temporal_input, flow_scale=12.0, residual=residual)


def test_dual_shapes_and_copy_last_frame_at_init():
    m = _dual("flow").eval()
    clip, flow = torch.rand(2, 4, 3, 64, 64), torch.randn(2, 3, 2, 32, 32)
    with torch.no_grad():
        y = m(clip, flow)
    assert y.shape == (2, 3, 64, 64) and torch.allclose(y, clip[:, -1], atol=1e-6)


def test_dual_flow_requirement_and_diff_mode():
    clip = torch.rand(2, 4, 3, 64, 64)
    with pytest.raises(ValueError):
        _dual("flow").eval()(clip)                                        # flow model without flow -> clear error
    with torch.no_grad():
        assert _dual("diff").eval()(clip).shape == (2, 3, 64, 64)         # frame differences need no flow
        assert _dual("both").eval()(clip, torch.randn(2, 3, 2, 32, 32)).shape == (2, 3, 64, 64)
        assert _dual("flow", residual=False).eval()(clip, torch.randn(2, 3, 2, 32, 32)).shape == (2, 3, 64, 64)


def test_build_model_guards_missing_flow():
    from vad.models import build_model
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.model.name, cfg.data.flow_mode = "dual_branch", "none"
    with pytest.raises(ValueError):
        build_model(cfg)
    cfg.model.temporal_input = "diff"
    assert build_model(cfg) is not None


def test_dual_trains_both_branches_and_uses_flow():
    from vad.lit_module import VADLitModule
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.model.name, cfg.model.embed_dim, cfg.model.depth, cfg.model.heads, cfg.model.mlp_ratio = "dual_branch", 32, 1, 4, 2.0
    cfg.model.depth_temporal, cfg.model.depth_fusion = 1, 1
    cfg.data.clip_len, cfg.data.img_size, cfg.data.flow_mode = 4, 64, "precomputed"
    lit = VADLitModule(cfg)
    opt = torch.optim.SGD(lit.parameters(), lr=0.1)
    batch = {"clip": torch.rand(4, 4, 3, 64, 64), "target": torch.rand(4, 3, 64, 64), "flow": torch.randn(4, 3, 2, 32, 32)}
    for _ in range(3):
        opt.zero_grad()
        loss = lit.training_step(batch, 0)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    assert lit.model.temporal.patch_embed.weight.grad.abs().sum() > 0         # motion branch receives gradient
    assert lit.model.spatial.patch_embed.weight.grad.abs().sum() > 0          # appearance branch receives gradient
    lit.eval()
    with torch.no_grad():
        y1 = lit.model(batch["clip"], batch["flow"])
        y0 = lit.model(batch["clip"], torch.zeros_like(batch["flow"]))
    assert (y1 - y0).abs().max().item() > 0                                   # the flow input actually matters
    assert not _has_dictconfig(dict(lit.hparams))


def test_set_keep_attn_reaches_every_attention_module():
    m = _dual("diff").eval()
    m.set_keep_attn(True)
    with torch.no_grad():
        m(torch.rand(1, 4, 3, 64, 64))
    stored = [mod.last_attn for mod in m.modules() if hasattr(mod, "keep_attn")]
    assert len(stored) >= 6 and all(a is not None for a in stored)


def test_benchmark_feeds_flow_to_flow_models():
    from vad.eval.runner import benchmark
    b = benchmark(_dual("flow"), 4, 64, torch.device("cpu"), n=1, flow_size=32)
    assert b["params"] > 0 and b["fps"] > 0
