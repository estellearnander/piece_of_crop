"""Features and hyperparameter tuning for the random forest baseline (notebook 03;
the features are also used by notebook 07).

The features follow the course notebook on image classification with classical
models, which describes each image by the mean and standard deviation of its
colour values (ImageClassificationML.pdf, pages 5 and 6).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid

from .data import BAND_NAMES, normalise

RGB_INDICES = [1, 2, 3]  # B02, B03, B04
RGB_FEATURES = [f"mean_{BAND_NAMES[i]}" for i in RGB_INDICES] + [f"std_{BAND_NAMES[i]}" for i in RGB_INDICES]
MS_FEATURES = [f"mean_{band}" for band in BAND_NAMES] + [f"std_{band}" for band in BAND_NAMES] + ["ndvi_mean", "ndvi_std"]


def extract_rf_features(images, band_stats: pd.DataFrame, batch_size: int = 1000):
    """Return (RGB features, 13-band features), one row per patch.

    RGB: mean and standard deviation of B02, B03 and B04 (6 features).
    13 bands: mean and standard deviation of all 13 bands, plus the mean and
    standard deviation of NDVI = (B08 - B04) / (B08 + B04) (28 features).
    The bands are clipped and normalised with the training statistics first.
    The patches are processed in batches, because all 27,000 do not fit in memory at once.
    """
    rgb_features, ms_features = [], []
    for start in range(0, len(images), batch_size):
        raw = np.asarray(images[start:start + batch_size], dtype=np.float32)
        bands = normalise(raw, band_stats)
        band_mean = bands.mean(axis=(2, 3))
        band_std = bands.std(axis=(2, 3))
        ndvi = (raw[:, 7] - raw[:, 3]) / (raw[:, 7] + raw[:, 3] + 1e-6)
        rgb_features.append(np.column_stack((band_mean[:, RGB_INDICES], band_std[:, RGB_INDICES])))
        ms_features.append(np.column_stack((band_mean, band_std, ndvi.mean(axis=(1, 2)), ndvi.std(axis=(1, 2)))))
    return np.vstack(rgb_features), np.vstack(ms_features)


def tune_random_forest(X_train, y_train, X_val, y_val, param_grid: dict, seed: int = 0,
                       n_estimators: int = 100):
    """Try every combination in param_grid and score it by macro-F1 on the validation set.

    The test patches are never used here.
    Returns (table of all combinations from best to worst, best parameters).
    """
    rows = []
    for params in ParameterGrid(param_grid):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        predicted = model.predict(X_val)
        rows.append({"params": params,
                     "val_accuracy": accuracy_score(y_val, predicted),
                     "val_macro_f1": f1_score(y_val, predicted, average="macro")})
    table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    return table, table.loc[0, "params"]
