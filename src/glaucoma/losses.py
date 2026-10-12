"""Training loss: binary cross-entropy plus soft Dice, averaged over the output channels.

Cross-entropy judges every pixel on its own, and the cup is a small fraction of the image, so a model can
score well by predicting "not cup" everywhere. Soft Dice judges overlap of the whole region, which punishes
that. Using both gives stable early training and good final overlap.
"""

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - soft Dice, computed per image and per channel, then averaged. ``target`` is float 0/1."""
    prob = torch.sigmoid(logits)
    dims = (2, 3)
    inter = (prob * target).sum(dims)
    total = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (total + eps)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equal-weight sum of BCE and soft Dice. ``logits`` and ``target`` are ``(B, C, H, W)``."""
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)
