"""Train one stage of the pipeline.

    python -m glaucoma.train locator   --epochs 30
    python -m glaucoma.train segmenter --epochs 40

The locator learns to find the disc in a whole photo. The segmenter learns to outline disc and cup in a
disc-centred crop. Each epoch the model is scored on the validation photos, and the best epoch is saved.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from glaucoma import data
from glaucoma.losses import segmentation_loss
from glaucoma.model import UNet
from glaucoma.pipeline import pick_device, save_checkpoint

STAGE_CLASSES = {"locator": 1, "segmenter": 2}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def validation_dice(model: UNet, loader: DataLoader, device: torch.device) -> float:
    """Mean per-image Dice over all output channels, on the model's own input resolution."""
    model.eval()
    scores = []
    for images, targets in loader:
        pred = model(images.to(device)) > 0
        target = targets.to(device) > 0.5
        inter = (pred & target).sum(dim=(2, 3)).float()
        total = pred.sum(dim=(2, 3)).float() + target.sum(dim=(2, 3)).float()
        scores.append(torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(total)).flatten())
    return float(torch.cat(scores).mean())


def train_stage(
    stage: str,
    train_records: list[data.Record],
    val_records: list[data.Record],
    cache_dir: Path,
    out_path: Path,
    epochs: int,
    batch_size: int = 8,
    lr: float = 3e-4,
    seed: int = 0,
    pretrained: bool = True,
    device: torch.device | None = None,
) -> float:
    """Train and save the best checkpoint to ``out_path``. Returns the best validation Dice."""
    seed_everything(seed)
    device = device or pick_device()
    data.build_cache(train_records + val_records, cache_dir)
    train_set = data.CachedDataset(train_records, cache_dir, stage, train=True)
    val_set = data.CachedDataset(val_records, cache_dir, stage, train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=len(train_set) > batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = UNet(pretrained=pretrained, n_classes=STAGE_CLASSES[stage]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best = -1.0
    print(f"{stage}: {len(train_set)} train / {len(val_set)} val photos on {device}")
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = segmentation_loss(model(images), targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        val = validation_dice(model, val_loader, device)
        marker = ""
        if val > best:
            best = val
            save_checkpoint(model, out_path, stage=stage, epoch=epoch, val_dice=val)
            marker = "  (saved)"
        print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  val dice {val:.4f}  {time.time() - started:.0f}s{marker}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=list(STAGE_CLASSES))
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="use only N training photos (smoke test)")
    args = parser.parse_args()

    split = data.split_records(data.load_records(args.data_root), seed=args.seed)
    train, val = split["train"], split["val"]
    if args.limit:
        train, val = train[: args.limit], val[: max(2, args.limit // 4)]
    best = train_stage(
        args.stage, train, val, Path(args.cache), Path(args.out) / f"{args.stage}.pt",
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )
    print(f"best validation dice: {best:.4f}")


if __name__ == "__main__":
    main()
