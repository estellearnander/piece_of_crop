"""Evaluation of the saved test predictions (notebook 06).

The insurer cost matrix, the critical confusions, loading and checking the 18
prediction files, and a bootstrap that resamples whole spatial blocks.
"""

import numpy as np
import pandas as pd

from .data import CLASS_NAMES

CROPS = {"AnnualCrop", "PermanentCrop"}
AGRICULTURAL = {"AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation"}
NOT_FARMED = {"Forest", "Highway", "Industrial", "Residential"}

MODEL_FILES = {"RandomForest": "rf", "SmallCNN": "cnn", "ResNet18": "resnet"}


def insurer_cost_matrix() -> np.ndarray:
    """Cost of predicting column j when the true class is row i, in severity points.

    These are the weights committed on 22 November 2026, before the current results:
    9  agricultural land confused with Forest or built-up land
    8  a crop class confused with another agricultural class
    6  Pasture confused with HerbaceousVegetation
    2  every other mistake, 0 a correct prediction
    """
    n = len(CLASS_NAMES)
    costs = np.full((n, n), 2.0)
    for i, actual in enumerate(CLASS_NAMES):
        for j, predicted in enumerate(CLASS_NAMES):
            if i == j:
                costs[i, j] = 0.0
            elif actual in AGRICULTURAL and predicted in AGRICULTURAL:
                costs[i, j] = 8.0 if (actual in CROPS or predicted in CROPS) else 6.0
            elif {actual, predicted} & AGRICULTURAL and {actual, predicted} & NOT_FARMED:
                costs[i, j] = 9.0
    return costs


def listed_critical_confusions() -> np.ndarray:
    """The earlier list of critical confusions (22 November), kept for the sensitivity analysis.

    A crop confused with any agricultural class, or a crop or Pasture confused with
    Forest or built-up land. Unlike cost >= 8, it leaves out HerbaceousVegetation
    confused with Forest or built-up land.
    """
    n = len(CLASS_NAMES)
    listed = np.zeros((n, n), dtype=bool)
    for i, actual in enumerate(CLASS_NAMES):
        for j, predicted in enumerate(CLASS_NAMES):
            if i == j:
                continue
            pair = {actual, predicted}
            if pair & CROPS and pair <= AGRICULTURAL:
                listed[i, j] = True
            if pair & (CROPS | {"Pasture"}) and pair & NOT_FARMED:
                listed[i, j] = True
    return listed


def load_predictions(predictions_dir, test_rows: pd.DataFrame, seeds=(0, 1, 2)) -> dict:
    """Read all 18 test prediction files, keyed by (model, input, seed).

    Every file must hold the test patches of split.csv in the same order with the
    same labels; otherwise the comparison would not be paired, so it stops.
    """
    predictions = {}
    for model, prefix in MODEL_FILES.items():
        for input_type in ("rgb", "13band"):
            for seed in seeds:
                path = predictions_dir / f"{prefix}_{input_type}_seed{seed}.csv"
                if not path.exists():
                    raise FileNotFoundError(f"{path.name} is missing: run notebooks 03 to 05 first")
                table = pd.read_csv(path)
                if not (np.array_equal(table["file"], test_rows["file"])
                        and np.array_equal(table["y_true"], test_rows["label"])):
                    raise ValueError(f"{path.name} does not match the test rows of split.csv")
                predictions[(model, input_type, seed)] = table
    return predictions


def block_bootstrap_ci(values, blocks, n_bootstrap, rng, level=0.95):
    """Interval for the mean of per-patch values, resampling whole spatial blocks.

    Patches in one block are neighbours and not independent, so a block is drawn
    with all its patches. Returns (low, high).
    """
    _, block_index = np.unique(blocks, return_inverse=True)
    block_sums = np.bincount(block_index, weights=values)
    block_sizes = np.bincount(block_index)
    n_blocks = len(block_sums)
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        drawn = rng.integers(0, n_blocks, size=n_blocks)
        means[b] = block_sums[drawn].sum() / block_sizes[drawn].sum()
    tail = (1 - level) / 2
    return np.quantile(means, tail), np.quantile(means, 1 - tail)
