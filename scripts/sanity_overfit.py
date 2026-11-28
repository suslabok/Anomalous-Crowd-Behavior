"""Can this model beat the trivial 'copy the last frame' baseline? And at which learning rate?

Trains from scratch on a handful of REAL clips (it should be able to memorise them) and compares the loss with the
copy baseline. Give several learning rates (comma separated) to scan them in one go.

    python scripts/sanity_overfit.py data.dataset=ped2 model.name=unet --amp --lr 1e-4,1e-3
    python scripts/sanity_overfit.py data.dataset=ped2 model.name=spatial_vit model.head_init=small --amp --lr 1e-4,3e-4,1e-3,3e-3 --steps 600
    python scripts/sanity_overfit.py data.dataset=ped2 model.name=dual_branch data.flow_mode=precomputed model.head_init=small --amp --lr 3e-4,1e-3
Options: --steps 400  --n 32 (clips)  --amp (16-bit mixed precision, as in train.py)  --lr default: train.lr

skill = 1 - loss / copy_loss: 0 means no better than copying the last frame, 1 means perfect.
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


def verdict(history, copy):
    """history: [(step, loss)] sampled every 50 steps."""
    final = 1 - history[-1][1] / copy
    best = max(1 - l / copy for _, l in history)
    recent = [l for _, l in history[-3:]]
    falling = len(recent) == 3 and recent[0] > 0 and (recent[0] - recent[-1]) / recent[0] > 0.05
    if final > 0.2:
        return final, best, "learns well (clearly below copy-last-frame)"
    if final > 0.05:
        return final, best, "beats copy-last-frame"
    if falling:
        return final, best, "still converging: not below copy yet but falling (needs more steps or a higher lr)"
    return final, best, "STUCK at or above the copy baseline"


def run(cfg, batch, dev, lr, steps, use_amp, verbose):
    set_seed(cfg.seed)
    model = build_model(cfg).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def loss_of(pred):
        return F.mse_loss(pred, batch["target"]) + cfg.train.gdl_weight * gradient_loss(pred, batch["target"])

    copy = loss_of(batch["clip"][:, -1]).item()
    history = []
    for step in range(steps + 1):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            pred = model(batch["clip"], batch.get("flow"))
        loss = loss_of(pred.float())
        if step % 50 == 0:
            history.append((step, loss.item()))
            if verbose:
                print(f"  step {step:4d}  loss {loss.item():.5f}  skill {1 - loss.item() / copy:+.3f}")
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(opt)
        scaler.update()
    params = sum(p.numel() for p in model.parameters())
    return history, copy, params, (scaler.get_scale() if use_amp else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", default=None, help="one value or comma-separated list")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--amp", action="store_true")
    args, overrides = ap.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and dev.type == "cuda"
    lrs = [float(x) for x in args.lr.split(",")] if args.lr else [float(cfg.train.lr)]

    set_seed(cfg.seed)
    train, _, _ = build_train_val(cfg, get_paths(cfg))
    items = [train[i] for i in torch.linspace(0, len(train) - 1, args.n).long().tolist()]   # spread over all videos
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in items[0]}

    rows = []
    for lr in lrs:
        print(f"\nmodel={cfg.model.name} head_init={cfg.model.get('head_init', '-')} lr={lr:g} amp={use_amp} "
              f"clips={args.n} steps={args.steps}")
        history, copy, params, scale = run(cfg, batch, dev, lr, args.steps, use_amp, verbose=len(lrs) == 1)
        final, best, text = verdict(history, copy)
        rows.append((lr, history[-1][1], final, best, text))
        print(f"  params {params / 1e6:.2f}M, copy-last-frame loss {copy:.5f}"
              + (f", AMP loss scale {scale:g}" if scale else ""))
    print("\nSUMMARY (skill 0 = no better than copying the last frame)")
    print(f"{'lr':>8} {'final loss':>11} {'final skill':>12} {'best skill':>11}  verdict")
    for lr, loss, final, best, text in rows:
        print(f"{lr:>8g} {loss:>11.5f} {final:>+12.3f} {best:>+11.3f}  {text}")


if __name__ == "__main__":
    main()
