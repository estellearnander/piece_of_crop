"""Cost matrices, critical errors, loading saved predictions and the paired
block bootstrap. Used by notebook 06.

The cost weights are severity points (0 for a correct label, 2 to 9 for a
mistake). They are scenario assumptions, not euros or real insurance losses.
"""

import numpy as np
import pandas as pd

from .data import CLASS_NAMES

CROPS = {"AnnualCrop", "PermanentCrop"}
AGRICULTURAL = {"AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation"}
BUILT = {"Highway", "Industrial", "Residential"}
NATURAL = {"Forest", "HerbaceousVegetation", "Pasture", "River", "SeaLake"}

MODEL_FILES = {"RandomForest": "rf", "SmallCNN": "cnn", "ResNet18": "resnet"}


def make_cost_matrices() -> dict:
    """Insurer and planner cost matrices, rows = actual class, columns = predicted class.

    Every mistake costs 2 by default.
    Insurer: confusions between agricultural classes cost 8 if a crop class is
    involved and 6 otherwise; agricultural versus built-up or Forest costs 9.
    Planner: built-up versus natural costs 9; crops versus Forest or water costs 7.
    """
    n = len(CLASS_NAMES)
    insurer = np.full((n, n), 2.0)
    planner = np.full((n, n), 2.0)
    np.fill_diagonal(insurer, 0.0)
    np.fill_diagonal(planner, 0.0)
    for i, actual in enumerate(CLASS_NAMES):
        for j, predicted in enumerate(CLASS_NAMES):
            if i == j:
                continue
            if actual in AGRICULTURAL and predicted in AGRICULTURAL:
                insurer[i, j] = 8.0 if actual in CROPS or predicted in CROPS else 6.0
            if ((actual in AGRICULTURAL and predicted in BUILT | {"Forest"}) or
                    (predicted in AGRICULTURAL and actual in BUILT | {"Forest"})):
                insurer[i, j] = 9.0
            if (actual in BUILT and predicted in NATURAL) or (predicted in BUILT and actual in NATURAL):
                planner[i, j] = 9.0
            if ((actual in CROPS and predicted in {"Forest", "River", "SeaLake"}) or
                    (predicted in CROPS and actual in {"Forest", "River", "SeaLake"})):
                planner[i, j] = 7.0
    return {"insurer": insurer, "planner": planner}


def is_critical(true_class: str, predicted_class: str) -> bool:
    """True for the confusions an insurer cares most about.

    A crop confused with another agricultural class, or crop or pasture
    confused with built-up land or Forest.
    """
    if true_class == predicted_class:
        return False
    if true_class in CROPS and predicted_class in AGRICULTURAL:
        return True
    if predicted_class in CROPS and true_class in AGRICULTURAL:
        return True
    if ((true_class in CROPS | {"Pasture"} and predicted_class in BUILT | {"Forest"}) or
            (predicted_class in CROPS | {"Pasture"} and true_class in BUILT | {"Forest"})):
        return True
    return False


def critical_matrix() -> np.ndarray:
    """10 x 10 boolean matrix, True where (actual, predicted) is a critical confusion."""
    return np.array([[is_critical(t, p) for p in CLASS_NAMES] for t in CLASS_NAMES], dtype=bool)


def cost_scenarios(cost_matrices: dict, critical: np.ndarray) -> dict:
    """The five cost assumptions of the sensitivity analysis."""
    all_equal = np.ones((10, 10))
    np.fill_diagonal(all_equal, 0.0)
    return {
        "equal_errors": all_equal,
        "insurer_baseline": cost_matrices["insurer"],
        "insurer_critical_half": cost_matrices["insurer"] * np.where(critical, 0.5, 1.0),
        "insurer_critical_double": cost_matrices["insurer"] * np.where(critical, 2.0, 1.0),
        "planner_baseline": cost_matrices["planner"],
    }


def load_predictions(predictions_dir, reference: pd.DataFrame, seeds=(0, 1, 2)):
    """Read every saved test prediction CSV and check it against the test split.

    reference: the test rows of split.csv (columns file and label).
    Returns (dict keyed by (model, input, seed), list of missing file names).
    A file with the wrong size, unknown classes or labels that differ from
    split.csv raises an error instead of being compared.
    """
    label_ids = set(range(len(CLASS_NAMES)))
    predictions, missing = {}, []
    for model, prefix in MODEL_FILES.items():
        for input_type in ("rgb", "13band"):
            for seed in seeds:
                path = predictions_dir / f"{prefix}_{input_type}_seed{seed}.csv"
                if not path.exists():
                    missing.append(path.name)
                    continue
                df = pd.read_csv(path)
                required = {"file", "y_true", "y_pred"}
                if not required.issubset(df.columns):
                    raise ValueError(f"{path.name} is missing columns {required - set(df.columns)}")
                if len(df) != len(reference) or df["file"].duplicated().any():
                    raise ValueError(f"{path.name}: wrong test size or duplicate file names")
                df["y_true"] = df["y_true"].astype(int)
                df["y_pred"] = df["y_pred"].astype(int)
                if not set(df["y_true"]) <= label_ids or not set(df["y_pred"]) <= label_ids:
                    raise ValueError(f"{path.name}: class IDs outside 0..9")
                aligned = reference[["file", "label"]].merge(df, on="file", validate="one_to_one", sort=False)
                if len(aligned) != len(reference) or not np.array_equal(aligned["label"], aligned["y_true"]):
                    raise ValueError(f"{path.name}: test labels or files differ from split.csv")
                predictions[(model, input_type, seed)] = aligned.drop(columns="label")
    return predictions, missing


def paired_block_bootstrap(per_patch_difference, groups, n_bootstraps, rng):
    """95% interval for a mean difference, resampling whole spatial blocks.

    per_patch_difference: one value per test patch (13-band cost minus RGB cost,
    averaged over seeds). groups: the block of each patch. Neighbouring patches
    in one block are not independent, so the block is the unit that is resampled.
    """
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    group_totals = np.bincount(group_index, weights=per_patch_difference)
    group_sizes = np.bincount(group_index)
    boot = np.empty(n_bootstraps)
    for b in range(n_bootstraps):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        boot[b] = group_totals[sampled].sum() / group_sizes[sampled].sum()
    return np.quantile(boot, 0.025), np.quantile(boot, 0.975), len(unique_groups)
