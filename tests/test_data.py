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
