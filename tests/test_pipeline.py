import json

import numpy as np
import torch

from glaucoma import data, evaluate, pipeline, train
from glaucoma.model import UNet


def untrained_models():
    torch.manual_seed(0)
    locator = UNet(pretrained=False, n_classes=1).eval()
    segmenter = UNet(pretrained=False, n_classes=2).eval()
    return locator, segmenter


def test_predict_masks_returns_full_size_masks_with_cup_inside_disc(fake_root):
    record = data.load_records(fake_root)[0]
    image = data.read_image(record.image_path)
    locator, segmenter = untrained_models()
    disc, cup = pipeline.predict_masks(image, locator, segmenter, torch.device("cpu"))
    assert disc.shape == cup.shape == image.shape[:2]
    assert disc.dtype == bool and cup.dtype == bool
    assert not (cup & ~disc).any()


def test_disc_centre_falls_back_to_photo_centre_when_nothing_found(fake_root):
    record = data.load_records(fake_root)[0]
    image = data.read_image(record.image_path)
    locator, _ = untrained_models()
    for p in locator.parameters():
        p.data.zero_()
    locator.head.bias.data.fill_(-10)  # predicts "no disc" everywhere
    h, w = image.shape[:2]
    assert pipeline.find_disc_centre(image, locator, torch.device("cpu")) == (h / 2, w / 2)


def test_segment_crop_places_masks_near_the_given_centre(fake_root):
    record = data.load_records(fake_root)[0]
    image = data.read_image(record.image_path)
    _, segmenter = untrained_models()
    segmenter.head.bias.data.copy_(torch.tensor([10.0, 10.0]))  # predicts disc and cup everywhere in the crop
    disc, _ = pipeline.segment_crop(image, (100.0, 150.0), segmenter, torch.device("cpu"))
    ys, xs = np.nonzero(disc)
    size = data.crop_size_for(*image.shape[:2])
    assert ys.min() >= 0 and xs.min() >= 0
    assert (ys.max() - ys.min() + 1) == size and (xs.max() - xs.min() + 1) == size


def test_checkpoint_round_trip_preserves_outputs(fake_root, tmp_path):
    locator, _ = untrained_models()
    pipeline.save_checkpoint(locator, tmp_path / "locator.pt", stage="locator")
    loaded = pipeline.load_checkpoint(tmp_path / "locator.pt", torch.device("cpu"))
    x = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        assert torch.allclose(locator(x), loaded(x), atol=1e-6)


def test_train_then_evaluate_end_to_end_on_synthetic_photos(fake_root, tmp_path, monkeypatch):
    records = data.load_records(fake_root)
    split = data.split_records(records, val_fraction=0.34, seed=0)
    cache, ckpts = tmp_path / "cache", tmp_path / "ckpt"
    data.build_cache(records, cache)
    for stage in ("locator", "segmenter"):
        best = train.train_stage(
            stage, split["train"], split["val"], cache, ckpts / f"{stage}.pt",
            epochs=1, batch_size=2, pretrained=False, device=torch.device("cpu"),
        )
        assert 0.0 <= best <= 1.0
        assert (ckpts / f"{stage}.pt").exists()

    out = tmp_path / "results"
    monkeypatch.setattr(pipeline, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(evaluate, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate", "--data-root", str(fake_root), "--checkpoints", str(ckpts), "--out", str(out)],
    )
    evaluate.main()

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["groups"]["All"]["n_images"] == len(split["test"])
    for key in ("disc_dice", "cup_dice", "disc_iou", "cup_iou", "vcdr_mae"):
        assert 0.0 <= metrics["groups"]["All"][key] <= 1.0
    assert metrics["baseline"]["n_images"] == len(split["test"])
    assert len(metrics["per_image"]) == len(split["test"])
    for scores in metrics["per_image"].values():  # the committed file must not carry label-derived values
        assert not {"vcdr_true", "vcdr_pred"} & set(scores)
    full = json.loads((out / "per_image.json").read_text())
    assert all("vcdr_true" in v for v in full.values())
    assert (out / "table.md").read_text().startswith("| Photos |")
    assert "| Baseline: same vCDR" in (out / "table.md").read_text()
    assert (out / "bias.md").read_text().startswith("| True vCDR |")
    assert "Failure cases" in (out / "failure_cases.md").read_text()
    assert list((out / "figures").glob("*.png"))


def test_table_rows_match_the_metrics_they_were_built_from():
    groups = {"All": {"n_images": 10, "disc_dice": 0.95, "cup_dice": 0.9, "disc_iou": 0.9,
                      "cup_iou": 0.82, "vcdr_mae": 0.0312, "vcdr_corr": 0.7}}
    baseline = {"n_images": 10, "constant_vcdr": 0.57, "vcdr_mae": 0.0917}
    table = evaluate.render_table(groups, baseline)
    assert "| All | 10 | 0.950 | 0.900 | 0.900 | 0.820 | 0.031 | 0.700 |" in table
    assert "| Baseline: same vCDR (0.57) for every photo | 10 | – | – | – | – | 0.092 | – |" in table
