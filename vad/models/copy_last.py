import torch.nn as nn


class CopyLastPredictor(nn.Module):
    """No parameters: the 'prediction' of the next frame is simply the last input frame, so the anomaly score is
    how much the scene changed. Every trained model must beat this floor; a model whose scores equal this
    baseline's has learned nothing."""

    def forward(self, clip, flow=None):
        return clip[:, -1]
