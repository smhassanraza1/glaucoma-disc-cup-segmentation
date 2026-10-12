"""Score the trained pipeline on held-out photos and write the results tables.

    python -m glaucoma.evaluate                          # the test photos (ORIGA set B)
    python -m glaucoma.evaluate --split val --out runs/val   # the validation photos, for tuning decisions

Writes ``metrics.json`` (summary numbers and per-photo scores), ``per_image.json`` (per-photo detail including
the specialist's ratio; not committed), ``table.md`` (the table the README shows), ``bias.md`` (vCDR error by
true-vCDR range) and ``failure_cases.md`` (the worst photos, by ID). Metrics are computed on full-resolution
masks, one photo at a time, then averaged. Overlay figures go to ``figures/`` inside the output folder, which
is not committed because they contain dataset images.

Anything tuned (thresholds, post-processing, which epoch to keep) must be decided on ``--split val``. The test
photos are for one final look.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from glaucoma import data
from glaucoma.metrics import ImageScores, score_image, summarize
from glaucoma.pipeline import load_checkpoint, pick_device, predict_masks
from glaucoma.vcdr import vertical_cdr

COLUMNS = [
    ("Disc Dice", "disc_dice"), ("Cup Dice", "cup_dice"), ("Disc IoU", "disc_iou"),
    ("Cup IoU", "cup_iou"), ("vCDR MAE", "vcdr_mae"), ("vCDR corr", "vcdr_corr"),
]
BIN_EDGES = [(0.0, 0.45), (0.45, 0.60), (0.60, 0.75), (0.75, 1.01)]


def train_median_vcdr(records: list[data.Record]) -> float:
    """Median specialist vCDR over the training photos: the 'always guess the typical value' baseline."""
    return float(np.median([vertical_cdr(*data.load_mask(r.mask_path)) for r in records]))


def baseline_summary(per_image: dict[str, ImageScores], constant: float) -> dict[str, float]:
    errors = [abs(constant - s.vcdr_true) for s in per_image.values()]
    return {"n_images": len(errors), "constant_vcdr": constant, "vcdr_mae": float(np.mean(errors))}


def render_table(groups: dict[str, dict[str, float]], baseline: dict[str, float] | None = None) -> str:
    header = "| Photos | n | " + " | ".join(name for name, _ in COLUMNS) + " |"
    rule = "|---|---:|" + "---:|" * len(COLUMNS)
    rows = [
        f"| {label} | {int(m['n_images'])} | " + " | ".join(f"{m[key]:.3f}" for _, key in COLUMNS) + " |"
        for label, m in groups.items()
    ]
    if baseline:
        cells = ["–" if key != "vcdr_mae" else f"{baseline['vcdr_mae']:.3f}" for _, key in COLUMNS]
        rows.append(f"| Baseline: same vCDR ({baseline['constant_vcdr']:.2f}) for every photo | "
                    f"{int(baseline['n_images'])} | " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows]) + "\n"


def render_bias(per_image: dict[str, ImageScores]) -> str:
    """vCDR error split by the specialist's vCDR. A positive bias means the model guesses too high."""
    lines = ["| True vCDR | n | vCDR MAE | Bias (predicted - true) |", "|---|---:|---:|---:|"]
    for lo, hi in BIN_EDGES:
        group = [s for s in per_image.values() if lo <= s.vcdr_true < hi]
        if not group:
            continue
        errors = [s.vcdr_pred - s.vcdr_true for s in group]
        label = f"{lo:.2f} to {min(hi, 1.0):.2f}"
        lines.append(f"| {label} | {len(group)} | {np.mean(np.abs(errors)):.3f} | {np.mean(errors):+.3f} |")
    return "\n".join(lines) + "\n"


def render_failures(per_image: dict[str, ImageScores], k: int = 5) -> str:
    by_cup = sorted(per_image.items(), key=lambda kv: kv[1].cup_dice)[:k]
    by_vcdr = sorted(per_image.items(), key=lambda kv: -kv[1].vcdr_error)[:k]
    lines = ["# Failure cases", "", "vCDR error is predicted minus true: negative means the cup was underestimated.",
             "", f"Worst {k} photos by cup Dice:", ""]
    lines += [f"- photo {s}: cup Dice {v.cup_dice:.3f}, disc Dice {v.disc_dice:.3f}, "
              f"vCDR error {v.vcdr_pred - v.vcdr_true:+.3f}" for s, v in by_cup]
    lines += ["", f"Worst {k} photos by vCDR error:", ""]
    lines += [f"- photo {s}: vCDR error {v.vcdr_pred - v.vcdr_true:+.3f}, cup Dice {v.cup_dice:.3f}"
              for s, v in by_vcdr]
    return "\n".join(lines) + "\n"


def save_overlay(path: Path, image, disc, cup, true_disc, true_cup) -> None:
    """Prediction (tinted) and specialist (outline) on the photo, cropped around the disc."""
    ys, xs = np.nonzero(true_disc | disc)
    pad = 60
    y0, y1 = max(ys.min() - pad, 0), ys.max() + pad
    x0, x1 = max(xs.min() - pad, 0), xs.max() + pad
    view = image[y0:y1, x0:x1].copy()
    for mask, colour in ((disc, (0, 200, 0)), (cup, (220, 0, 0))):
        m = mask[y0:y1, x0:x1]
        view[m] = (0.55 * view[m] + 0.45 * np.array(colour)).astype(np.uint8)
    for mask, colour in ((true_disc, (255, 255, 0)), (true_cup, (0, 255, 255))):
        contours, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(view, contours, -1, colour, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(view, (512, 512)), cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--out", default="results")
    parser.add_argument("--limit", type=int, default=None, help="score only N photos (smoke test)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    device = pick_device()
    locator = load_checkpoint(Path(args.checkpoints) / "locator.pt", device)
    segmenter = load_checkpoint(Path(args.checkpoints) / "segmenter.pt", device)
    splits = data.split_records(data.load_records(args.data_root))
    records = splits[args.split]
    if args.limit:
        records = records[: args.limit]

    per_image: dict[str, ImageScores] = {}
    cases: dict[str, tuple] = {}
    for record in tqdm(records, desc="evaluating", unit="img"):
        image = data.read_image(record.image_path)
        true_disc, true_cup = data.load_mask(record.mask_path)
        disc, cup = predict_masks(image, locator, segmenter, device)
        per_image[record.stem] = score_image(disc, cup, true_disc, true_cup)
        if not args.no_figures:
            cases[record.stem] = (disc, cup, true_disc, true_cup)
            if len(cases) > 40:  # keep memory bounded: only photos that may be failures later
                worst = set(sorted(per_image, key=lambda s: per_image[s].cup_dice)[:5]) | set(
                    sorted(per_image, key=lambda s: -per_image[s].vcdr_error)[:5])
                cases = {s: v for s, v in cases.items() if s in worst}

    by_label = {r.stem: r.glaucoma for r in records}
    groups = {
        "All": summarize(list(per_image.values())),
        **{name: summarize([v for s, v in per_image.items() if by_label[s] == flag])
           for name, flag in (("Normal", 0), ("Glaucoma", 1)) if any(by_label[s] == flag for s in per_image)},
    }
    baseline = baseline_summary(per_image, train_median_vcdr(splits["train"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full = {s: {**v.__dict__, "vcdr_error": v.vcdr_error} for s, v in per_image.items()}
    shareable = ("disc_dice", "cup_dice", "disc_iou", "cup_iou", "vcdr_error")  # nothing that reveals a label
    (out / "metrics.json").write_text(json.dumps({
        "split": f"ORIGA {'set B (test)' if args.split == 'test' else 'set A validation photos'}",
        "device": str(device),
        "groups": groups,
        "baseline": baseline,
        "per_image": {s: {k: v[k] for k in shareable} for s, v in full.items()},
    }, indent=2))
    # The specialist's ratio for every photo is dataset label data, so this file is not committed.
    (out / "per_image.json").write_text(json.dumps(full, indent=2))
    (out / "table.md").write_text(render_table(groups, baseline))
    (out / "bias.md").write_text(render_bias(per_image))
    (out / "failure_cases.md").write_text(render_failures(per_image))

    if not args.no_figures:
        worst = sorted(per_image, key=lambda s: per_image[s].cup_dice)[:5] + sorted(
            per_image, key=lambda s: -per_image[s].vcdr_error)[:5]
        for stem in dict.fromkeys(worst):
            if stem in cases:
                record = next(r for r in records if r.stem == stem)
                save_overlay(out / "figures" / f"{stem}.png", data.read_image(record.image_path), *cases[stem])

    print(render_table(groups, baseline))
    print(render_bias(per_image))


if __name__ == "__main__":
    main()
