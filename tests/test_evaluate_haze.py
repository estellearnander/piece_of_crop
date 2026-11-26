from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from piece_of_crop import (
    CLASS_NAMES,
    block_bootstrap_ci,
    insurer_cost_matrix,
    listed_critical_confusions,
    BAND_NAMES,
    WAVELENGTHS_NM,
    add_haze,
    haze_share,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_cost_matrix_has_zero_diagonal_and_is_symmetric():
    costs = insurer_cost_matrix()
    assert costs.shape == (10, 10)
    assert np.all(np.diag(costs) == 0)
    assert np.all(costs[~np.eye(10, dtype=bool)] >= 2)
    assert np.array_equal(costs, costs.T)


def test_cost_matrix_is_the_committed_table():
    """The weights were fixed on 22 November, before the current results, and must not change."""
    saved = pd.read_csv(RESULTS / "insurer_cost_matrix.csv", index_col=0)
    assert list(saved.columns) == list(CLASS_NAMES)
    assert np.array_equal(saved.to_numpy(), insurer_cost_matrix())


def test_cost_levels_of_example_confusions():
    costs = pd.DataFrame(insurer_cost_matrix(), index=CLASS_NAMES, columns=CLASS_NAMES)
    assert costs.loc["AnnualCrop", "PermanentCrop"] == 8
    assert costs.loc["Pasture", "HerbaceousVegetation"] == 6
    assert costs.loc["HerbaceousVegetation", "Forest"] == 9
    assert costs.loc["River", "Highway"] == 2


def test_earlier_critical_list_is_inside_cost_8_or_more():
    listed = listed_critical_confusions()
    assert np.all(insurer_cost_matrix()[listed] >= 8)
    assert listed.sum() < (insurer_cost_matrix() >= 8).sum()


@pytest.mark.parametrize("profile", ["uniform", "wavelength"])
def test_zero_haze_changes_nothing(profile):
    raw = np.random.default_rng(0).integers(0, 4000, (2, 13, 8, 8)).astype(np.float32)
    assert np.array_equal(add_haze(raw, 0.0, profile, np.full(13, 5000.0)), raw)


def test_wavelength_haze_is_strongest_in_blue_and_weakest_in_swir():
    share = haze_share(0.5, "wavelength")
    order = np.argsort(WAVELENGTHS_NM)
    assert np.all(np.diff(share[order]) < 0)  # shorter wavelength, more haze
    assert np.isclose(share[BAND_NAMES.index("B03")], 0.5, atol=0.02)  # 560 nm is close to 550 nm
    assert np.allclose(haze_share(0.2, "uniform"), 0.2)


def test_heavy_haze_moves_values_towards_the_airlight():
    raw = np.zeros((13, 4, 4), dtype=np.float32)
    airlight = np.full(13, 1000.0)
    hazy = add_haze(raw, 0.999, "uniform", airlight)
    assert np.all(hazy > 990)


def test_block_bootstrap_interval_contains_a_constant_difference():
    low, high = block_bootstrap_ci(np.full(30, -0.5), np.repeat(np.arange(6), 5), 200, np.random.default_rng(0))
    assert np.isclose(low, -0.5) and np.isclose(high, -0.5)


def test_block_bootstrap_interval_is_wider_when_blocks_differ():
    rng = np.random.default_rng(0)
    values = np.repeat([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 5)
    low, high = block_bootstrap_ci(values, np.repeat(np.arange(6), 5), 500, rng)
    assert low < 0.5 < high
