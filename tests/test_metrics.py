import numpy as np
import pytest

from glaucoma.metrics import dice, iou, score_image, summarize
from glaucoma.vcdr import vertical_cdr, vertical_diameter


def disc_and_cup(size=100, disc_r=30, cup_r=15, centre=(50, 50)):
    yy, xx = np.mgrid[:size, :size]
    dist = np.hypot(yy - centre[0], xx - centre[1])
    return dist <= disc_r, dist <= cup_r


def test_dice_and_iou_known_values():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2, :] = True  # 8 pixels
    b[:3, :] = True  # 12 pixels, overlap 8
    assert dice(a, b) == pytest.approx(2 * 8 / 20)
    assert iou(a, b) == pytest.approx(8 / 12)


def test_dice_identical_and_disjoint_and_empty():
    a = np.zeros((4, 4), bool)
    a[0] = True
    b = np.zeros((4, 4), bool)
    b[3] = True
    assert dice(a, a) == 1.0
    assert dice(a, b) == 0.0
    assert dice(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 1.0


def test_vertical_diameter_is_tallest_column():
    mask = np.zeros((10, 10), bool)
    mask[2:5, 3] = True  # column of 3
    mask[1:8, 6] = True  # column of 7
    assert vertical_diameter(mask) == 7.0
    assert vertical_diameter(np.zeros((5, 5), bool)) == 0.0


def test_vertical_cdr_of_concentric_circles_is_radius_ratio():
    disc, cup = disc_and_cup(disc_r=30, cup_r=15)
    assert vertical_cdr(disc, cup) == pytest.approx(0.5, abs=0.05)


def test_vertical_cdr_is_cup_over_disc_not_disc_over_cup():
    disc, cup = disc_and_cup()
    assert vertical_cdr(disc, cup) < 1.0


def test_vertical_cdr_ignores_width():
    # A wide, flat cup has the same vertical diameter as a thin one.
    disc = np.zeros((40, 80), bool)
    disc[10:30, 10:70] = True  # height 20
    flat_cup = np.zeros_like(disc)
    flat_cup[18:22, 15:65] = True  # height 4, very wide
    assert vertical_cdr(disc, flat_cup) == pytest.approx(4 / 20, abs=1e-4)


def test_score_image_perfect_prediction():
    disc, cup = disc_and_cup()
    scores = score_image(disc, cup, disc, cup)
    assert scores.disc_dice == scores.cup_dice == 1.0
    assert scores.vcdr_error == pytest.approx(0.0)


def test_summarize_averages_per_image_and_reports_vcdr_mae():
    disc, cup = disc_and_cup(cup_r=15)
    _, bigger_cup = disc_and_cup(cup_r=24)
    perfect = score_image(disc, cup, disc, cup)
    off = score_image(disc, bigger_cup, disc, cup)
    result = summarize([perfect, off])
    assert result["n_images"] == 2
    assert result["cup_dice"] == pytest.approx((1.0 + off.cup_dice) / 2)
    assert result["vcdr_mae"] == pytest.approx(off.vcdr_error / 2)
