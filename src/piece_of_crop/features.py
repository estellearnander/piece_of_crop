"""Tabular features and hyperparameter tuning for the random forest baseline
(notebook 03; the features are also used by notebook 07).

Each band is clipped and standardised with the training statistics from
band_stats.csv, then summarised by its mean and standard deviation over the patch.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid

from .data import BAND_NAMES

RGB_INDICES = (1, 2, 3)  # B02, B03, B04 (feature order used for the saved RF results)


def extract_rf_features(
    cache: np.ndarray,
    band_stats: pd.DataFrame,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract RGB and multispectral tabular features from identical patches.

    Each spectral band contributes a spatial mean and standard deviation after
    clipping/standardization with TRAIN-ONLY statistics from notebook 02.
    In the multispectral variant, also add mean/std NDVI computed from raw
    B08 (NIR) and B04 (red) reflectance. NDVI is excluded from RGB features.
    """
    required = {"band", "clip_low", "clip_high", "mean", "std"}
    if not required.issubset(band_stats.columns):
        raise ValueError(f"band_stats.csv is missing {sorted(required - set(band_stats.columns))}")
    stats = band_stats.set_index("band").reindex(BAND_NAMES)
    if stats.isna().any().any():
        raise ValueError("band_stats.csv missing values/bands in expected BAND_NAMES order")

    low = stats["clip_low"].to_numpy(dtype=np.float32)[None, :, None, None]
    high = stats["clip_high"].to_numpy(dtype=np.float32)[None, :, None, None]
    mean = stats["mean"].to_numpy(dtype=np.float32)[None, :, None, None]
    std = stats["std"].to_numpy(dtype=np.float32)[None, :, None, None]
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("band_stats standard deviations must be positive and finite")

    rgb_columns = [f"{stat}_{BAND_NAMES[i]}" for stat in ("mean", "std")
                   for i in RGB_INDICES]
    ms_columns = (
        [f"mean_{band}" for band in BAND_NAMES]
        + [f"std_{band}" for band in BAND_NAMES]
        + ["ndvi_mean", "ndvi_std"]
    )

    n = len(cache)
    X_rgb = np.empty((n, len(rgb_columns)), dtype=np.float32)
    X_ms = np.empty((n, len(ms_columns)), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        raw = np.asarray(cache[start:end], dtype=np.float32)
        x = (np.clip(raw, low, high) - mean) / std
        channel_mean = x.mean(axis=(2, 3))
        channel_std = x.std(axis=(2, 3))
        X_rgb[start:end] = np.column_stack((
            channel_mean[:, RGB_INDICES], channel_std[:, RGB_INDICES]
        ))
        ndvi = (raw[:, 7] - raw[:, 3]) / (raw[:, 7] + raw[:, 3] + 1e-6)
        X_ms[start:end] = np.column_stack((
            channel_mean,
            channel_std,
            ndvi.mean(axis=(1, 2)),
            ndvi.std(axis=(1, 2)),
        ))
        if end % 3000 < batch_size or end == n:
            print(f"Feature extraction: {end:,} / {n:,} images")

    if not np.isfinite(X_rgb).all() or not np.isfinite(X_ms).all():
        raise ValueError("Some extracted features are NaN/inf; inspect input TIFFs and stats")
    return X_rgb, X_ms, rgb_columns, ms_columns


def tune_random_forest(X_train, y_train, X_val, y_val, param_grid: dict, seed: int = 0,
                       n_estimators: int = 200):
    """Try every combination in param_grid and score it on the validation set only.

    Each combination is trained on the training patches and scored by macro-F1
    on the validation patches; the test patches are never used here.
    Returns (table of all combinations sorted from best to worst, best parameters).
    """
    rows, candidates = [], []
    for params in ParameterGrid(param_grid):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        predicted = model.predict(X_val)
        candidates.append(params)
        rows.append({**{key: str(value) for key, value in params.items()},
                     "val_accuracy": accuracy_score(y_val, predicted),
                     "val_macro_f1": f1_score(y_val, predicted, average="macro")})
    table = pd.DataFrame(rows)
    best = int(table["val_macro_f1"].to_numpy().argmax())
    return table.sort_values("val_macro_f1", ascending=False).reset_index(drop=True), candidates[best]

