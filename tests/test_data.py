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
