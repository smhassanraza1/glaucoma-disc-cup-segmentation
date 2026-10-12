"""Two-stage inference: find the disc, crop around it, outline the disc and cup, clean up the masks.

Stage 1 (locator) looks at the whole photo at low resolution and returns a rough disc mask, whose centre
tells stage 2 where to look. Stage 2 (segmenter) sees only a square crop around that centre, at much higher
detail than the whole photo would allow, and outlines the disc and the cup. The two masks are pasted back
into a full-size canvas and cleaned.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from glaucoma import data
from glaucoma.model import UNet
from glaucoma.postprocess import clean_masks


def pick_device() -> torch.device:
    """CUDA on Colab, Metal (MPS) on an Apple-silicon Mac, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: UNet, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "n_classes": model.head.out_channels, **meta}, path)


def load_checkpoint(path: str | Path, device: torch.device) -> UNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet(pretrained=False, n_classes=ckpt["n_classes"])
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval()


@torch.no_grad()
def find_disc_centre(image: np.ndarray, locator: UNet, device: torch.device) -> tuple[float, float]:
    """Centre ``(row, column)`` of the disc, in the coordinates of the full-size photo."""
    h, w = image.shape[:2]
    x = data.to_tensor(data.make_locator_input(image))[None].to(device)
    disc = (torch.sigmoid(locator(x))[0, 0] > 0.5).cpu().numpy()
    if not disc.any():  # nothing found: fall back to the photo's centre
        return h / 2, w / 2
    cy, cx = data.centroid(disc)
    return cy * h / data.LOCATOR_HW[0], cx * w / data.LOCATOR_HW[1]


@torch.no_grad()
def segment_crop(
    image: np.ndarray, centre: tuple[float, float], segmenter: UNet, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Outline disc and cup in a crop around ``centre``. Returns full-size boolean ``(disc, cup)``."""
    h, w = image.shape[:2]
    size = data.crop_size_for(h, w)
    y0, x0 = data.crop_origin(*centre, h, w, size)
    crop = image[y0 : y0 + size, x0 : x0 + size]
    resized = cv2.resize(data.make_cached_crop(crop), (data.SEG_SIZE,) * 2, interpolation=cv2.INTER_LINEAR)
    probs = torch.sigmoid(segmenter(data.to_tensor(resized)[None].to(device)))
    probs = F.interpolate(probs, size=(size, size), mode="bilinear", align_corners=False)[0].cpu().numpy()
    disc = np.zeros((h, w), dtype=bool)
    cup = np.zeros((h, w), dtype=bool)
    disc[y0 : y0 + size, x0 : x0 + size] = probs[0] > 0.5
    cup[y0 : y0 + size, x0 : x0 + size] = probs[1] > 0.5
    return disc, cup


def predict_masks(
    image: np.ndarray, locator: UNet, segmenter: UNet, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline on one RGB photo. Returns cleaned full-size boolean ``(disc, cup)``."""
    centre = find_disc_centre(image, locator, device)
    disc, cup = segment_crop(image, centre, segmenter, device)
    return clean_masks(disc, cup)
