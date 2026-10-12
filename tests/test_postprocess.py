import numpy as np

from glaucoma.postprocess import clean_masks, largest_component


def square(size, top, left, side):
    mask = np.zeros((size, size), bool)
    mask[top : top + side, left : left + side] = True
    return mask


def test_largest_component_drops_stray_blob():
    mask = square(40, 5, 5, 15) | square(40, 30, 30, 3)
    out = largest_component(mask)
    assert out.sum() == 15 * 15
    assert not out[30:33, 30:33].any()


def test_largest_component_of_empty_mask_is_empty():
    assert not largest_component(np.zeros((8, 8), bool)).any()


def test_clean_masks_fills_holes_and_removes_islands():
    disc = square(60, 10, 10, 30)
    disc[20:23, 20:23] = False  # hole
    disc |= square(60, 52, 52, 3)  # island
    cup = square(60, 18, 18, 12)
    disc_out, cup_out = clean_masks(disc, cup)
    assert disc_out[21, 21]
    assert not disc_out[53, 53]
    assert cup_out.sum() == 12 * 12


def test_clean_masks_forces_cup_inside_disc():
    disc = square(60, 10, 10, 20)
    cup = square(60, 5, 5, 15)  # sticks out of the disc
    disc_out, cup_out = clean_masks(disc, cup)
    assert not (cup_out & ~disc_out).any()


def test_clean_masks_handles_empty_prediction():
    empty = np.zeros((16, 16), bool)
    disc_out, cup_out = clean_masks(empty, empty)
    assert not disc_out.any() and not cup_out.any()
