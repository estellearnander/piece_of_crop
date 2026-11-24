from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from piece_of_crop import (
    CLASS_NAMES,
    critical_matrix,
    make_cost_matrices,
    paired_block_bootstrap,
    simulate_haze,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_cost_matrices_have_zero_diagonal_and_positive_errors():
    for matrix in make_cost_matrices().values():
        assert matrix.shape == (10, 10)
        assert np.all(np.diag(matrix) == 0)
        assert np.all(matrix[~np.eye(10, dtype=bool)] > 0)


@pytest.mark.parametrize("name", ["insurer", "planner"])
def test_cost_matrices_match_the_saved_tables(name):
    saved = pd.read_csv(RESULTS / f"{name}_cost_matrix.csv", index_col=0)
    assert list(saved.columns) == list(CLASS_NAMES)
    assert np.array_equal(saved.to_numpy(), make_cost_matrices()[name])


def test_critical_confusions_have_the_highest_insurer_costs():
    insurer = make_cost_matrices()["insurer"]
    assert insurer[critical_matrix()].min() >= 8


@pytest.mark.parametrize("profile", ["uniform", "wavelength"])
def test_zero_haze_changes_nothing(profile):
    raw = np.random.default_rng(0).integers(0, 4000, (2, 13, 8, 8)).astype(np.float32)
    assert np.array_equal(simulate_haze(raw, 0.0, profile, np.full(13, 5000.0)), raw)


def test_full_uniform_haze_gives_the_bright_reference():
    raw = np.zeros((13, 4, 4), dtype=np.float32)
    reference = np.arange(13, dtype=np.float32) * 100
    hazy = simulate_haze(raw, 1.0, "uniform", reference)
    assert np.allclose(hazy[:, 0, 0], reference)


def test_block_bootstrap_interval_contains_a_constant_difference():
    low, high, n_groups = paired_block_bootstrap(np.full(30, -0.5), np.repeat(np.arange(6), 5),
                                                 200, np.random.default_rng(0))
    assert n_groups == 6 and np.isclose(low, -0.5) and np.isclose(high, -0.5)
