import numpy as np
import pytest

from glaucoma import data
from glaucoma.vcdr import vertical_cdr


def test_load_records_reads_labels_and_sets(fake_root):
    records = data.load_records(fake_root)
    assert [r.stem for r in records] == [f"{i:03d}" for i in range(1, 9)]
    assert {r.subset for r in records} == {"A", "B"}
    assert [r.glaucoma for r in records] == [0, 0, 1, 0, 0, 1, 0, 0]


def test_excluded_photo_is_skipped(fake_root, monkeypatch):
    monkeypatch.setattr(data, "EXCLUDED", frozenset({"003"}))
    assert "003" not in [r.stem for r in data.load_records(fake_root)]


def test_load_mask_uses_mask_gt0_as_disc_and_gt1_as_cup(fake_root):
    record = data.load_records(fake_root)[0]
    disc, cup = data.load_mask(record.mask_path)
    assert disc.dtype == bool and cup.dtype == bool
    assert cup.sum() > 0 and (cup & ~disc).sum() == 0
    assert 0 < vertical_cdr(disc, cup) < 1


def test_split_puts_set_b_in_test_and_keeps_sets_disjoint(fake_root):
    split = data.split_records(data.load_records(fake_root), val_fraction=0.25, seed=1)
    assert {r.subset for r in split["test"]} == {"B"}
    assert {r.subset for r in split["train"] + split["val"]} == {"A"}
    stems = lambda k: {r.stem for r in split[k]}
    assert not (stems("train") & stems("val"))
    assert stems("train") | stems("val") == {"001", "003", "005", "007"}
    assert len(split["val"]) >= 1


def test_split_is_reproducible(fake_root):
    records = data.load_records(fake_root)
    a = data.split_records(records, seed=3)
    b = data.split_records(records, seed=3)
    assert [r.stem for r in a["val"]] == [r.stem for r in b["val"]]


def test_crop_origin_stays_inside_photo():
    assert data.crop_origin(5, 5, 100, 200, 40) == (0, 0)
    assert data.crop_origin(99, 199, 100, 200, 40) == (60, 160)
    assert data.crop_origin(50, 100, 100, 200, 40) == (30, 80)


def test_crop_size_scales_with_width():
    assert data.crop_size_for(2048, 3072) == 800
    assert data.crop_size_for(1024, 1536) == 400
    assert data.crop_size_for(100, 100) <= 100


def test_cache_has_expected_shapes_and_disc_in_crop_centre(fake_root, tmp_path):
    cache = tmp_path / "cache"
    record = data.load_records(fake_root)[0]
    data.build_cache([record], cache)
    with np.load(cache / "001.npz") as z:
        assert z["small"].shape == (*data.LOCATOR_HW, 3)
        assert z["small_disc"].shape == data.LOCATOR_HW
        assert z["crop"].shape == (data.CACHE_CROP, data.CACHE_CROP, 3)
        assert z["crop_disc"].any() and z["crop_cup"].any()
        cy, cx = data.centroid(z["crop_disc"] > 0)
    assert abs(cy - data.CACHE_CROP / 2) < 30 and abs(cx - data.CACHE_CROP / 2) < 30


def test_cache_rejects_mismatched_mask(fake_root, tmp_path):
    from scipy import io as sio

    record = data.load_records(fake_root)[0]
    sio.savemat(str(record.mask_path), {"mask": np.zeros((10, 10), np.uint8)})
    with pytest.raises(ValueError, match="does not match"):
        data.cache_record(record, tmp_path / "cache")


@pytest.mark.parametrize("stage,channels,size", [("locator", 1, data.LOCATOR_HW), ("segmenter", 2, (256, 256))])
@pytest.mark.parametrize("train", [True, False])
def test_dataset_items(fake_root, tmp_path, stage, channels, size, train):
    records = data.load_records(fake_root)[:3]
    data.build_cache(records, tmp_path / "cache")
    ds = data.CachedDataset(records, tmp_path / "cache", stage=stage, train=train)
    image, target = ds[0]
    assert len(ds) == 3
    assert image.shape == (3, *size) and target.shape == (channels, *size)
    assert set(np.unique(target.numpy())) <= {0.0, 1.0}
    assert target.sum() > 0
