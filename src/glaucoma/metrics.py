"""Segmentation metrics: Dice, IoU, and the absolute error of the vertical cup-to-disc ratio.

Dice and IoU both score how much a predicted region overlaps the specialist's region, from 0 (no overlap) to
1 (identical). Dice = 2 * overlap / (size of prediction + size of truth). IoU = overlap / union. They rank
models the same way, but IoU is always the lower of the two for the same pair of masks.

Each metric is computed per image and then averaged over images.
"""

from dataclasses import dataclass

import numpy as np

from glaucoma.vcdr import vertical_cdr


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice overlap of two boolean masks. Two empty masks count as a perfect match."""
    pred, target = pred.astype(bool), target.astype(bool)
    total = pred.sum() + target.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, target).sum() / total)


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection over union of two boolean masks. Two empty masks count as a perfect match."""
    pred, target = pred.astype(bool), target.astype(bool)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / union)


@dataclass(frozen=True)
class ImageScores:
    disc_dice: float
    cup_dice: float
    disc_iou: float
    cup_iou: float
    vcdr_pred: float
    vcdr_true: float

    @property
    def vcdr_error(self) -> float:
        return abs(self.vcdr_pred - self.vcdr_true)


def score_image(
    pred_disc: np.ndarray, pred_cup: np.ndarray, true_disc: np.ndarray, true_cup: np.ndarray
) -> ImageScores:
    """Score one image. Each disc mask includes its cup."""
    return ImageScores(
        disc_dice=dice(pred_disc, true_disc),
        cup_dice=dice(pred_cup, true_cup),
        disc_iou=iou(pred_disc, true_disc),
        cup_iou=iou(pred_cup, true_cup),
        vcdr_pred=vertical_cdr(pred_disc, pred_cup),
        vcdr_true=vertical_cdr(true_disc, true_cup),
    )


def summarize(scores: list[ImageScores]) -> dict[str, float]:
    """Average per-image scores. ``vcdr_mae`` is the mean absolute vCDR error and
    ``vcdr_corr`` the Pearson correlation between predicted and true vCDR."""
    if not scores:
        raise ValueError("no scores to summarize")
    pred = np.array([s.vcdr_pred for s in scores])
    true = np.array([s.vcdr_true for s in scores])
    varies = len(scores) > 2 and pred.std() > 0 and true.std() > 0
    return {
        "n_images": len(scores),
        "disc_dice": float(np.mean([s.disc_dice for s in scores])),
        "cup_dice": float(np.mean([s.cup_dice for s in scores])),
        "disc_iou": float(np.mean([s.disc_iou for s in scores])),
        "cup_iou": float(np.mean([s.cup_iou for s in scores])),
        "vcdr_mae": float(np.mean([s.vcdr_error for s in scores])),
        "vcdr_corr": float(np.corrcoef(pred, true)[0, 1]) if varies else float("nan"),
    }
