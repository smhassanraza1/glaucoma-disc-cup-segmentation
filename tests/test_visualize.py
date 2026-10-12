import numpy as np

from glaucoma.visualize import CUP_COLOUR, DISC_COLOUR, describe, overlay


def masks():
    disc = np.zeros((40, 40), bool)
    disc[10:30, 10:30] = True
    cup = np.zeros_like(disc)
    cup[15:25, 15:25] = True
    return disc, cup


def test_overlay_outlines_masks_and_does_not_modify_input():
    image = np.full((40, 40, 3), 100, np.uint8)
    disc, cup = masks()
    out = overlay(image, disc, cup)
    assert (image == 100).all()  # input untouched
    assert (out[0, 0] == 100).all()  # outside the masks
    assert (out[20, 10] == DISC_COLOUR).all()  # disc edge
    assert (out[20, 15] == CUP_COLOUR).all()  # cup edge
    assert (out[12, 20] == 100).all()  # disc interior is left as is


def test_describe_reports_ratio_or_missing_disc():
    disc, cup = masks()
    assert describe(disc, cup) == "Vertical cup-to-disc ratio: 0.50"
    empty = np.zeros_like(disc)
    assert describe(empty, empty) == "No optic disc found in this photo."
