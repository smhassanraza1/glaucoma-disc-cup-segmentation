"""A tiny fake ORIGA dataset, drawn in code, so the tests never need the real data."""

import csv

import cv2
import numpy as np
import pytest
from scipy import io as sio

H, W = 384, 576  # small photos, same 2:3 shape as ORIGA


def draw_photo_and_mask(seed: int):
    rng = np.random.default_rng(seed)
    cy, cx = H // 2 + rng.integers(-15, 15), W // 2 + rng.integers(-30, 30)
    yy, xx = np.mgrid[:H, :W]
    dist = np.hypot(yy - cy, xx - cx)
    disc_r, cup_r = 50 + rng.integers(0, 10), 22 + rng.integers(0, 8)
    mask = np.zeros((H, W), np.uint8)
    mask[dist <= disc_r] = 1
    mask[dist <= cup_r] = 2
    photo = np.full((H, W, 3), (40, 90, 160), np.uint8)
    photo[mask == 1] = (230, 200, 120)
    photo[mask == 2] = (250, 240, 200)
    photo = cv2.GaussianBlur(photo, (9, 9), 0)
    return photo, mask


@pytest.fixture()
def fake_root(tmp_path):
    """Eight photos: stems 001-008, sets A (odd) and B (even), glaucoma on every third."""
    images = tmp_path / "ORIGA" / "ORIGA" / "Images"
    masks = tmp_path / "ORIGA" / "ORIGA" / "Semi-automatic-annotations"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    rows = []
    for i in range(1, 9):
        stem = f"{i:03d}"
        photo, mask = draw_photo_and_mask(i)
        cv2.imwrite(str(images / f"{stem}.jpg"), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
        sio.savemat(str(masks / f"{stem}.mat"), {"mask": mask})
        rows.append(
            {"Filename": f"{stem}.jpg", "ExpCDR": 0.4, "Eye": "OD",
             "Set": "A" if i % 2 else "B", "Glaucoma": int(i % 3 == 0)}
        )
    with open(tmp_path / "glaucoma.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path
