import math


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    """LR multiplier: linear warm-up for `warmup` steps, then cosine decay to 0 at `total`."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, p))))
