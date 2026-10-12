"""ORIGA data loading, splitting, caching and augmentation.

ORIGA is 650 fundus photos (3072x2048) with one expert disc/cup annotation each, stored as MATLAB ``.mat``
files holding a ``mask`` array: 0 = background, 1 = disc rim, 2 = cup. The disc is every non-zero pixel and
the cup is ``mask > 1``. A ``glaucoma.csv`` lists each photo's diagnosis, eye side and a set label (A or B).

The model works in two stages, because the disc covers only about 1.6% of a photo:
  1. The locator sees the whole photo, shrunk to 256x384, and finds the disc.
  2. The segmenter sees a square crop around that disc and outlines the disc and the cup.

Decoding full-size JPEGs every epoch would be slow, so each photo is turned once into a small whole-photo
copy and a disc-centred crop, and those are cached as ``.npz`` files.

Expected layout under the data root::

    ORIGA/ORIGA/Images/001.jpg ...
    ORIGA/ORIGA/Semi-automatic-annotations/001.mat ...
    glaucoma.csv
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from scipy import io as sio
from torch.utils.data import Dataset
from tqdm import tqdm

LOCATOR_HW = (256, 384)  # whole photo, height x width
SEG_SIZE = 256  # segmenter input, square
CACHE_CROP = 384  # cached crop size; augmentation then resizes to SEG_SIZE
CROP_FRACTION = 800 / 3072  # 800 px on a 3072 px wide photo
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Photo 359 is 3085 px wide but its mask is 3072 px wide, so the two cannot be aligned.
EXCLUDED = frozenset({"359"})


@dataclass(frozen=True)
class Record:
    stem: str
    image_path: Path
    mask_path: Path
    glaucoma: int
    subset: str  # "A" or "B", from glaucoma.csv


def load_records(root: str | Path) -> list[Record]:
    """List the usable ORIGA photos that have a mask and a CSV row."""
    root = Path(root)
    origa = root / "ORIGA" / "ORIGA"
    with open(root / "glaucoma.csv", newline="") as f:
        rows = {r["Filename"].rsplit(".", 1)[0]: r for r in csv.DictReader(f)}
    records = []
    for image_path in sorted((origa / "Images").glob("*.jpg")):
        stem = image_path.stem
        mask_path = origa / "Semi-automatic-annotations" / f"{stem}.mat"
        if stem in EXCLUDED or stem not in rows or not mask_path.exists():
            continue
        row = rows[stem]
        records.append(Record(stem, image_path, mask_path, int(row["Glaucoma"]), row["Set"]))
    return records


def split_records(
    records: list[Record], val_fraction: float = 0.15, seed: int = 0
) -> dict[str, list[Record]]:
    """Set B is the test set. Set A is split into train and validation, keeping the glaucoma ratio."""
    rng = np.random.default_rng(seed)
    set_a = [r for r in records if r.subset == "A"]
    train, val = [], []
    for label in (0, 1):
        group = [r for r in set_a if r.glaucoma == label]
        order = rng.permutation(len(group))
        n_val = round(val_fraction * len(group))
        val += [group[i] for i in order[:n_val]]
        train += [group[i] for i in order[n_val:]]
    return {"train": train, "val": val, "test": [r for r in records if r.subset == "B"]}


def read_image(path: str | Path) -> np.ndarray:
    """Read a photo as an RGB uint8 array."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an ORIGA ``.mat`` annotation as boolean ``(disc, cup)`` masks. The disc includes the cup."""
    mask = sio.loadmat(str(path))["mask"]
    return mask > 0, mask > 1


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def crop_size_for(height: int, width: int) -> int:
    """Side of the disc-centred crop, scaled to the photo's width."""
    return max(32, min(height, width, round(width * CROP_FRACTION)))


def crop_origin(cy: float, cx: float, height: int, width: int, size: int) -> tuple[int, int]:
    """Top-left corner of a ``size`` square centred on (cy, cx), moved inside the photo if needed."""
    y0 = round(cy - size / 2)
    x0 = round(cx - size / 2)
    return min(max(y0, 0), height - size), min(max(x0, 0), width - size)


def make_locator_input(image: np.ndarray) -> np.ndarray:
    """Whole photo shrunk to the locator's input size."""
    h, w = LOCATOR_HW
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def make_cached_crop(crop: np.ndarray) -> np.ndarray:
    """Disc-centred crop shrunk to the cached size. Training and inference share this step."""
    return cv2.resize(crop, (CACHE_CROP, CACHE_CROP), interpolation=cv2.INTER_AREA)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """RGB uint8 HxWx3 to a normalised float tensor 3xHxW."""
    x = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def cache_record(record: Record, cache_dir: Path) -> Path:
    """Write the small whole-photo copy and the disc-centred crop for one photo, once."""
    out = cache_dir / f"{record.stem}.npz"
    if out.exists():
        return out
    image = read_image(record.image_path)
    disc, cup = load_mask(record.mask_path)
    if disc.shape != image.shape[:2]:
        raise ValueError(f"{record.stem}: mask {disc.shape} does not match image {image.shape[:2]}")
    h, w = disc.shape
    lh, lw = LOCATOR_HW
    cy, cx = centroid(disc)
    size = crop_size_for(h, w)
    y0, x0 = crop_origin(cy, cx, h, w, size)
    window = (slice(y0, y0 + size), slice(x0, x0 + size))
    nearest = cv2.INTER_NEAREST
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        small=make_locator_input(image),
        small_disc=cv2.resize(disc.astype(np.uint8), (lw, lh), interpolation=nearest),
        crop=make_cached_crop(image[window]),
        crop_disc=cv2.resize(disc[window].astype(np.uint8), (CACHE_CROP,) * 2, interpolation=nearest),
        crop_cup=cv2.resize(cup[window].astype(np.uint8), (CACHE_CROP,) * 2, interpolation=nearest),
    )
    return out


def build_cache(records: list[Record], cache_dir: str | Path) -> None:
    cache_dir = Path(cache_dir)
    for record in tqdm(records, desc="caching", unit="img"):
        cache_record(record, cache_dir)


def locator_transform(train: bool) -> A.Compose:
    if not train:
        return A.Compose([A.NoOp()])
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]
    )


def segmenter_transform(train: bool) -> A.Compose:
    resize = A.Resize(SEG_SIZE, SEG_SIZE)
    if not train:
        return A.Compose([resize])
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1), rotate=(-20, 20), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            resize,
        ]
    )


class CachedDataset(Dataset):
    """Cached photos for one stage: ``"locator"`` (disc only) or ``"segmenter"`` (disc and cup)."""

    def __init__(self, records: list[Record], cache_dir: str | Path, stage: str, train: bool):
        if stage not in ("locator", "segmenter"):
            raise ValueError(f"unknown stage: {stage}")
        self.stage = stage
        self.transform = locator_transform(train) if stage == "locator" else segmenter_transform(train)
        self.items = []
        for record in records:
            with np.load(Path(cache_dir) / f"{record.stem}.npz") as z:
                if stage == "locator":
                    self.items.append((z["small"], [z["small_disc"]]))
                else:
                    self.items.append((z["crop"], [z["crop_disc"], z["crop_cup"]]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, masks = self.items[index]
        out = self.transform(image=image, masks=masks)
        target = np.stack([m.astype(np.float32) for m in out["masks"]])
        return to_tensor(out["image"]), torch.from_numpy(target)
