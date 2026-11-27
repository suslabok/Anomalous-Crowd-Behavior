"""Usage:
    python train.py                         # default config
    python train.py --smoke                 # 2 epochs on dummy data
    python train.py data.dataset=avenue train.max_epochs=50   # OmegaConf overrides
Resumes automatically from the last checkpoint on Drive if one exists.
"""
import argparse
import inspect
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from vad.lit_module import VADLitModule
from vad.utils.paths import get_paths
from vad.utils.seed import set_seed
from vad.data.dummy import DummyClips


def get_loaders(cfg, paths, smoke):
    kw = dict(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
              persistent_workers=cfg.data.num_workers > 0)
    if smoke:   # dummy data, no dataset needed
        ds = lambda n: DummyClips(n, cfg.data.clip_len, 64)
        return DataLoader(ds(64), shuffle=True, **kw), DataLoader(ds(16), **kw)
    from vad.data.datasets import build_train_val
    # Validation clips come from held-out NORMAL training videos, never from test videos.
    train, val, val_ids = build_train_val(cfg, paths)
    print(f"[data] {cfg.data.dataset}: {len(train)} train clips, "
          f"{len(val) if val else 0} val clips from held-out normal videos {val_ids}")
    return (DataLoader(train, shuffle=True, drop_last=True, pin_memory=True, **kw),
            DataLoader(val, pin_memory=True, **kw) if val else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    args, overrides = ap.parse_known_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    if args.smoke:
        cfg.train.max_epochs = 2
    set_seed(cfg.seed)
    paths = get_paths(cfg)

    # Same run name => same checkpoint folder => training RESUMES. Pass run_name=<new> for a fresh run.
    run_name = (cfg.get("run_name") or f"{cfg.data.dataset}_{cfg.model.name}") + ("_smoke" if args.smoke else "")
    ckpt_dir = paths.checkpoints / run_name

    # Keep last.ckpt (used to resume) and the checkpoint with the lowest validation loss.
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, save_last=True, save_top_k=1,
                              monitor="val/loss", mode="min",
                              every_n_epochs=cfg.train.ckpt_every_n_epochs)
    # 16-bit mixed precision saves GPU memory on Colab (the smoke test uses 32-bit).
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="auto", devices=1,
        precision=cfg.train.precision if not args.smoke else 32,
        gradient_clip_val=cfg.train.grad_clip,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=TensorBoardLogger(paths.logs, name=run_name),
        callbacks=[ckpt_cb, LearningRateMonitor("epoch")],
    )

    # If an earlier run left last.ckpt on Drive, continue from it (this is what survives Colab disconnects).
    last = ckpt_dir / "last.ckpt"
    resume = str(last) if last.exists() else None
    print(f"[train] run={run_name} resume={resume}")

    train_dl, val_dl = get_loaders(cfg, paths, args.smoke)
    fit_kw = {}
    if "weights_only" in inspect.signature(trainer.fit).parameters:
        fit_kw["weights_only"] = False   # our own checkpoints; torch>=2.6 defaults to True
    trainer.fit(VADLitModule(cfg), train_dl, val_dl, ckpt_path=resume, **fit_kw)
    skill = trainer.callback_metrics.get("val/skill")
    if skill is not None:
        print(f"[train] final val skill vs the 'copy the last frame' baseline: {float(skill):+.3f}")
        if float(skill) < 0.05:
            print("[warn] the model is barely better than copying the last frame, i.e. it has learned (almost) nothing. "
                  "Run scripts/sanity_overfit.py to find out why before trusting any evaluation.")


if __name__ == "__main__":
    main()
