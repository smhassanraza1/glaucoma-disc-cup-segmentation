"""Draw predicted disc and cup masks on a fundus photo, and describe the measurement for the demo."""

import numpy as np
from scipy import ndimage

from glaucoma.vcdr import vertical_cdr

DISC_COLOUR = (0, 255, 0)
CUP_COLOUR = (0, 128, 255)


def overlay(image: np.ndarray, disc: np.ndarray, cup: np.ndarray) -> np.ndarray:
    """Draw the disc outline in green and the cup outline in blue on an RGB photo. Returns a new array."""
    out = image.copy()
    width = max(2, image.shape[1] // 500)  # keeps the line visible on full-size photos
    for mask, colour in ((disc, DISC_COLOUR), (cup, CUP_COLOUR)):
        if mask.any():
            out[mask & ~ndimage.binary_erosion(mask, iterations=width)] = colour
    return out


def describe(disc: np.ndarray, cup: np.ndarray) -> str:
    """One-line summary of the measurement. Says so when no disc was found."""
    if not disc.any():
        return "No optic disc found in this photo."
    return f"Vertical cup-to-disc ratio: {vertical_cdr(disc, cup):.2f}"
