"""Mask clean-up applied to a network's raw output before measuring anything.

A network can leave small stray blobs away from the optic disc, or small holes inside it. Neither is
anatomically possible: there is one disc, it has no holes, and the cup sits inside it. This module removes
the blobs, fills the holes, and forces the cup inside the disc.
"""

import numpy as np
from scipy import ndimage


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob of a boolean mask."""
    mask = mask.astype(bool)
    labelled, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = ndimage.sum(mask, labelled, index=np.arange(1, count + 1))
    return labelled == (int(np.argmax(sizes)) + 1)


def clean_masks(disc: np.ndarray, cup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cleaned ``(disc, cup)``: one hole-free blob each, with the cup inside the disc."""
    disc = ndimage.binary_fill_holes(largest_component(disc))
    cup = ndimage.binary_fill_holes(largest_component(cup & disc if disc.any() else cup))
    return disc, cup & disc
