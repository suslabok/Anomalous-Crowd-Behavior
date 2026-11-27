"""Can this model beat the trivial 'copy the last frame' baseline AT ALL?

Trains on a handful of REAL clips (it should be able to memorise them) and prints the loss next to the copy baseline.
If the loss never drops clearly below the baseline, training is broken or the architecture cannot express the task.

    python scripts/sanity_overfit.py data.dataset=ped2 model.name=unet --amp
    python scripts/sanity_overfit.py data.dataset=ped2 model.name=spatial_vit --amp
    python scripts/sanity_overfit.py data.dataset=ped2 model.name=spatial_vit --amp model.head_init=small
    python scripts/sanity_overfit.py data.dataset=ped2 model.name=dual_branch data.flow_mode=precomputed --amp
Options: --steps 400  --lr <default: train.lr>  --n 32 (clips)  --amp (16-bit mixed precision, as in train.py)
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vad.data.datasets import build_train_val
from vad.models import build_model
from vad.models.losses import gradient_loss
from vad.utils.paths import get_paths
from vad.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--amp", action="store_true")
    args, overrides = ap.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and dev.type == "cuda"

    train, _, _ = build_train_val(cfg, get_paths(cfg))
    items = [train[i] for i in torch.linspace(0, len(train) - 1, args.n).long().tolist()]   # spread over all videos
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in items[0]}

    model = build_model(cfg).to(dev).train()
    params = sum(p.numel() for p in model.parameters())
    lr = args.lr or cfg.train.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def loss_of(pred):
        return F.mse_loss(pred, batch["target"]) + cfg.train.gdl_weight * gradient_loss(pred, batch["target"])

    copy = loss_of(batch["clip"][:, -1]).item()
    print(f"model={cfg.model.name} params={params / 1e6:.2f}M lr={lr:g} amp={use_amp} clips={args.n} "
          f"copy-last-frame loss={copy:.5f}")
    for step in range(args.steps + 1):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            pred = model(batch["clip"], batch.get("flow"))
        loss = loss_of(pred.float())
        if step % 50 == 0:
            print(f"  step {step:4d}  loss {loss.item():.5f}  skill {1 - loss.item() / copy:+.3f}")
        if step == args.steps:
            break
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(opt)
        scaler.update()
    final = 1 - loss.item() / copy
    if use_amp:
        print(f"  AMP loss scale at the end: {scaler.get_scale():g} (it collapses towards 1 when steps keep overflowing)")
    print("VERDICT:", "learns (clearly beats copy-last-frame)" if final > 0.2 else
          "DOES NOT LEARN: barely beats copy-last-frame even when memorising a few clips")


if __name__ == "__main__":
    main()
