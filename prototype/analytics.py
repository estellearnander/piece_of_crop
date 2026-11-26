"""Read only saved test outputs. Does not run training or require model weights."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MODEL_PREFIXES = {"Random Forest": "rf", "Small CNN": "cnn", "ResNet18": "resnet"}


def read_test_split(root: Path) -> pd.DataFrame:
    path = root / "results" / "split.csv"
    if not path.is_file():
        return pd.DataFrame()
    table = pd.read_csv(path)
    needed = {"file", "label", "class_name", "split"}
    if not needed.issubset(table.columns):
        return pd.DataFrame()
    return table[table["split"].eq("test")].reset_index(drop=True)


def read_verified_predictions(root: Path) -> tuple[dict, list[str]]:
    """Only accept results that match the CURRENT test split row-by-row.

    This prevents charts from mixing earlier split versions with new predictions.
    """
    split = read_test_split(root)
    warnings = []
    if split.empty:
        return {}, ["Missing or invalid results/split.csv; cannot verify saved predictions."]
    predictions = {}
    for model, prefix in MODEL_PREFIXES.items():
        for input_type in ("rgb", "13band"):
            for seed in (0, 1, 2):
                name = f"{prefix}_{input_type}_seed{seed}.csv"
                path = root / "results" / "predictions" / name
                if not path.is_file():
                    warnings.append(f"Missing: {name}")
                    continue
                try:
                    df = pd.read_csv(path)
                    needed = {"file", "y_true", "y_pred", *[f"p_{i}" for i in range(10)]}
                    if not needed.issubset(df.columns):
                        warnings.append(f"Invalid columns: {name}")
                        continue
                    if len(df) != len(split) or not np.array_equal(df["file"].to_numpy(), split["file"].to_numpy()) \
                       or not np.array_equal(df["y_true"].to_numpy(), split["label"].to_numpy()):
                        warnings.append(f"Split mismatch: {name} (excluded from charts)")
                        continue
                    if df["y_pred"].isna().any() or not df["y_pred"].between(0, 9).all():
                        warnings.append(f"Invalid predictions: {name}")
                        continue
                    predictions[(model, input_type, seed)] = df
                except Exception as e:
                    warnings.append(f"Cannot read {name}: {e}")
    return predictions, warnings


def metrics_table(predictions: dict, costs: np.ndarray) -> pd.DataFrame:
    rows = []
    for (model, input_type, seed), df in predictions.items():
        truth = df["y_true"].to_numpy(dtype=int)
        pred = df["y_pred"].to_numpy(dtype=int)
        cost = costs[truth, pred]
        rows.append({
            "model": model,
            "input": input_type,
            "seed": seed,
            "samples": len(df),
            "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, labels=list(range(10)), average="macro", zero_division=0)),
            "mean_severity": float(cost.mean()),
            "critical_errors": int((cost >= 8).sum()),
        })
    return pd.DataFrame(rows)


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    out = (metrics.groupby(["model", "input"], as_index=False)
           .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                mean_severity=("mean_severity", "mean"),
                critical_errors=("critical_errors", "mean"), seeds=("seed", "nunique")))
    return out


def critical_confusions(table: pd.DataFrame, costs: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    truth = table["y_true"].to_numpy(dtype=int)
    pred = table["y_pred"].to_numpy(dtype=int)
    rows = []
    for i in range(10):
        for j in range(10):
            if i != j:
                count = int(((truth == i) & (pred == j)).sum())
                if count:
                    rows.append({"Actual": class_names[i], "Predicted": class_names[j],
                                 "Count": count, "Severity points": float(costs[i, j]),
                                 "Total severity": float(costs[i, j] * count)})
    return pd.DataFrame(rows).sort_values("Total severity", ascending=False) if rows else pd.DataFrame()


HAZE_MODEL_NAMES = {"RandomForest": "Random Forest", "SmallCNN": "Small CNN", "ResNet18": "ResNet18"}


def read_haze_summary(root: Path, profile: str) -> pd.DataFrame:
    """Mean over seeds of the notebook 07 test-set results for one haze profile.

    The clear images (profile "none", strength 0) are the starting point of every curve.
    """
    path = root / "results" / "haze_runs.csv"
    if not path.is_file():
        return pd.DataFrame()
    runs = pd.read_csv(path)
    runs = runs[runs["profile"].isin(["none", profile])]
    out = (runs.groupby(["model", "input", "haze_550"], as_index=False)
           .agg(accuracy=("accuracy", "mean"), critical_error_rate=("critical_error_rate", "mean"),
                seeds=("seed", "nunique")))
    out["model"] = out["model"].map(HAZE_MODEL_NAMES)
    out["input"] = out["input"].map({"rgb": "RGB", "13band": "13 bands"})
    return out
