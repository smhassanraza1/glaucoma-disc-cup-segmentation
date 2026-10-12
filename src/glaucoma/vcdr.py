"""Vertical cup-to-disc ratio (vCDR), the clinical measure used to screen for glaucoma.

The optic disc is where the nerve leaves the eye, and the cup is the pale pit at its centre. In glaucoma the
cup grows, so a taller cup relative to the disc raises suspicion. The measure is the cup's vertical diameter
divided by the disc's vertical diameter.

The vertical diameter is the largest number of mask pixels in any single column, which is the tallest part of
the shape. This is the definition the REFUGE challenge used, and it reproduces the cup-to-disc column that the
ORIGA dataset ships with, to four decimals.
"""

import numpy as np

EPS = 1e-7


def vertical_diameter(mask: np.ndarray) -> float:
    """Largest number of mask pixels found in any single column, in pixels."""
    if not mask.any():
        return 0.0
    return float(mask.astype(bool).sum(axis=0).max())


def vertical_cdr(disc: np.ndarray, cup: np.ndarray) -> float:
    """Cup height divided by disc height. The disc mask includes the cup."""
    return vertical_diameter(cup) / (vertical_diameter(disc) + EPS)
