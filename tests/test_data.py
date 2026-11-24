import numpy as np
import pandas as pd

from piece_of_crop.data import (
    BAND_NAMES,
    RGB_BAND_INDICES,
    find_duplicates,
    pixel_fingerprint,
    to_rgb_image,
)


def test_band_names_and_rgb_positions():
    assert len(BAND_NAMES) == 13
    assert [BAND_NAMES[i] for i in RGB_BAND_INDICES] == ["B04", "B03", "B02"]


def test_identical_pixels_give_identical_fingerprints():
    a = np.ones((13, 64, 64), dtype=np.float32)
    b = a.copy()
    c = a.copy()
    c[0, 0, 0] = 2
    assert pixel_fingerprint(a) == pixel_fingerprint(b)
    assert pixel_fingerprint(a) != pixel_fingerprint(c)


def test_find_duplicates_returns_only_the_repeated_files():
    summary = pd.DataFrame({"path": ["x", "y", "z"], "fingerprint": ["h1", "h1", "h2"]})
    assert sorted(find_duplicates(summary)["path"]) == ["x", "y"]


def test_rgb_image_has_display_shape_and_range():
    bands = np.random.default_rng(0).uniform(0, 3000, (13, 64, 64)).astype(np.float32)
    image = to_rgb_image(bands)
    assert image.shape == (64, 64, 3)
    assert image.min() >= 0 and image.max() <= 1


from piece_of_crop.data import apply_buffer, normalise, select_bands, spatial_split


def _toy_patches():
    rng = np.random.default_rng(0)
    rows = []
    for block in range(40):
        lon, lat = block * 1.0, 45.0
        for i in range(10):
            rows.append({"file": f"f{block}_{i}", "label": i % 5,
                         "lon": lon + rng.uniform(0, 0.4), "lat": lat + rng.uniform(0, 0.4)})
    return pd.DataFrame(rows)


def test_spatial_split_keeps_each_file_and_block_in_one_split():
    split = spatial_split(_toy_patches(), block_size_deg=0.5, n_folds=7, seed=0)
    assert split["file"].is_unique
    assert set(split["split"]) == {"train", "val", "test"}
    assert (split.groupby("block")["split"].nunique() == 1).all()


def test_buffer_removes_training_patches_next_to_test_patches():
    table = pd.DataFrame({"lon": [0.0, 0.001, 5.0], "lat": [45.0, 45.0, 45.0],
                          "split": ["test", "train", "train"]})
    result = apply_buffer(table, buffer_km=1.0)
    assert list(result["split"]) == ["test", "buffer", "train"]


def test_normalise_gives_mean_zero_and_std_one_with_matching_stats():
    rng = np.random.default_rng(0)
    bands = rng.normal(1000, 200, (13, 64, 64)).astype(np.float32)
    stats = pd.DataFrame({"clip_low": [-1e9] * 13, "clip_high": [1e9] * 13,
                          "mean": bands.reshape(13, -1).mean(1), "std": bands.reshape(13, -1).std(1)})
    out = normalise(bands, stats)
    assert np.allclose(out.reshape(13, -1).mean(1), 0, atol=1e-3)
    assert np.allclose(out.reshape(13, -1).std(1), 1, atol=1e-3)


def test_select_bands_shapes():
    bands = np.zeros((13, 64, 64), dtype=np.float32)
    assert select_bands(bands, "rgb").shape == (3, 64, 64)
    assert select_bands(bands, "13band").shape == (13, 64, 64)


from pathlib import Path

import pytest

from piece_of_crop.data import EuroSATDataset, augment, load_cache, make_loaders, read_bands, read_split


def _toy_loader_inputs(n_per_split=6):
    rng = np.random.default_rng(0)
    n = 3 * n_per_split
    cache = rng.integers(0, 5000, (n, 13, 64, 64)).astype(np.uint16)
    split = pd.DataFrame({
        "cache_row": np.arange(n),
        "label": np.arange(n) % 10,
        "split": ["train"] * n_per_split + ["val"] * n_per_split + ["test"] * n_per_split,
    })
    stats = pd.DataFrame({"clip_low": [0.0] * 13, "clip_high": [5000.0] * 13,
                          "mean": [2500.0] * 13, "std": [1000.0] * 13})
    return split, cache, stats


@pytest.mark.parametrize("input_type, channels", [("rgb", 3), ("13band", 13)])
def test_loader_batch_shapes_and_labels(input_type, channels):
    split, cache, stats = _toy_loader_inputs()
    loaders = make_loaders(split, cache, stats, input_type, batch_size=4, seed=0)
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (4, channels, 64, 64)
    assert images.dtype.is_floating_point
    assert labels.min() >= 0 and labels.max() <= 9


def test_validation_patch_is_identical_every_time():
    split, cache, stats = _toy_loader_inputs()
    val = EuroSATDataset(split[split["split"] == "val"], cache, stats, "13band", augment_data=False)
    first, _ = val[0]
    second, _ = val[0]
    assert np.array_equal(first.numpy(), second.numpy())


def test_augment_only_moves_pixels():
    bands = np.random.default_rng(1).normal(size=(13, 64, 64)).astype(np.float32)
    moved = augment(bands, np.random.default_rng(2))
    assert moved.shape == bands.shape
    assert np.array_equal(np.sort(moved.ravel()), np.sort(bands.ravel()))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "eurosat_cache.npy"


@pytest.mark.skipif(not CACHE_PATH.exists(), reason="cache not built on this machine")
def test_cache_rows_match_the_geotiffs():
    split = read_split(PROJECT_ROOT / "results" / "split.csv")
    cache = load_cache(CACHE_PATH, split)
    for row in [0, 1234, len(split) - 1]:
        path = PROJECT_ROOT / "data" / "EuroSAT_MS" / split.loc[row, "class_name"] / split.loc[row, "file"]
        assert np.array_equal(cache[row].astype(np.float32), read_bands(path))


def test_get_cache_builds_when_missing_and_then_reuses(tmp_path):
    from pathlib import Path
    from piece_of_crop import get_cache, read_bands, read_split
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "EuroSAT_MS"
    if not data_dir.exists():
        pytest.skip("EuroSAT_MS not downloaded")
    split = read_split(root / "results" / "split.csv").head(3).copy()
    cache_path = tmp_path / "cache.npy"
    cache = get_cache(split, data_dir, cache_path)
    first = data_dir / split.loc[0, "class_name"] / split.loc[0, "file"]
    assert np.array_equal(cache[0].astype(np.float32), read_bands(first))
    modified = cache_path.stat().st_mtime_ns
    get_cache(split, data_dir, cache_path)
    assert cache_path.stat().st_mtime_ns == modified


def test_eval_loader_is_not_augmented_or_shuffled():
    import torch
    from piece_of_crop.data import make_eval_loader
    split, cache, stats = _toy_loader_inputs()
    loader = make_eval_loader(split, cache, stats, "13band", "train", batch_size=4)
    first = torch.cat([images for images, _ in loader])
    second = torch.cat([images for images, _ in loader])
    labels = torch.cat([labels for _, labels in loader])
    assert torch.equal(first, second)
    assert labels.tolist() == split.loc[split["split"] == "train", "label"].tolist()
