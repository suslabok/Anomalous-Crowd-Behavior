import torch


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 difference between spatial gradients of prediction and target (sharpness term of L_pred)."""
    dxp = (pred[..., :, 1:] - pred[..., :, :-1]).abs()
    dxt = (target[..., :, 1:] - target[..., :, :-1]).abs()
    dyp = (pred[..., 1:, :] - pred[..., :-1, :]).abs()
    dyt = (target[..., 1:, :] - target[..., :-1, :]).abs()
    return (dxp - dxt).abs().mean() + (dyp - dyt).abs().mean()
