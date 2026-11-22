"""Random Forest baseline helpers for the EuroSAT multispectral project.

Place this file at: src/piece_of_crop/features.py
The notebook imports it directly (no ``piece_of_crop`` package installation needed).

Assumptions from notebooks 01/02:
- results/split.csv rows define cache ordering.
- Cache stores uint16 arrays with shape (n_patches, 13, 64, 64).
- results/band_stats.csv was fit on TRAINING images only.
- Sentinel-2 bands follow B01..B12, B8A last, as validated in notebook 01.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

BAND_NAMES = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A",
)
CLASS_NAMES = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
)
RGB_INDICES = (1, 2, 3)  # B02, B03, B04; same 3 channels as notebook 02


def read_split(path: str | Path) -> pd.DataFrame:
    """Load the immutable spatial split saved by notebook 02 (do NOT reshuffle)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Get results/split.csv from the team's repository "
            "or run notebook 02 in the full project."
        )
    split = pd.read_csv(path)
    required = {"file", "class_name", "label", "split"}
    missing = required.difference(split.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if split["split"].isna().any() or not split["split"].isin(
        ["train", "val", "test", "buffer"]
    ).all():
        raise ValueError("Unexpected split values. Expected train/val/test/buffer.")
    if not {"train", "val", "test"}.issubset(set(split["split"])):
        raise ValueError("split.csv must contain train, val, and test examples.")
    split["label"] = split["label"].astype(int)
    return split.reset_index(drop=True)


def get_cache(
    split: pd.DataFrame, data_dir: str | Path, cache_path: str | Path
) -> np.ndarray:
    """Memory-map the shared cache; build it from raw TIFFs if necessary.

    Building the cache requires EuroSAT_MS/ and rasterio. The file order matches
    the rows of split.csv exactly, including buffer rows.
    """
    cache_path = Path(cache_path)
    data_dir = Path(data_dir)

    if not cache_path.is_file():
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Cache not found: {cache_path}\n"
                f"Raw images not found: {data_dir}\n"
                "The EuroSAT_MS image data and the ~2.9 GB cache are not stored "
                "in the notebooks. Download the EuroSAT multispectral GeoTIFF "
                "data into data/EuroSAT_MS/<class>/<image>.tif, or copy the "
                "existing data/eurosat_cache.npy from a teammate."
            )
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "The cache is missing, so rasterio is needed to read TIFFs. "
                "Install it with `python -m pip install rasterio` (using the "
                "same environment as your notebook)."
            ) from exc

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(".eurosat_cache_build.npy")
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            cache = np.lib.format.open_memmap(
                tmp_path, mode="w+", dtype=np.uint16,
                shape=(len(split), len(BAND_NAMES), 64, 64),
            )
            for idx, row in enumerate(split.itertuples(index=False)):
                tif = data_dir / row.class_name / row.file
                if not tif.is_file():
                    raise FileNotFoundError(f"Missing EuroSAT image: {tif}")
                with rasterio.open(tif) as src:
                    raw = src.read()
                if raw.shape != (len(BAND_NAMES), 64, 64):
                    raise ValueError(f"Wrong band/image shape for {tif}: {raw.shape}")
                cache[idx] = raw
                if (idx + 1) % 3000 == 0:
                    print(f"Cached {idx + 1:,} / {len(split):,} images")
            cache.flush()
            del cache
            os.replace(tmp_path, cache_path)
            print(f"Created {cache_path}")
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    expected = (len(split), len(BAND_NAMES), 64, 64)
    if cache.shape != expected:
        raise ValueError(
            f"Cache shape {cache.shape} != expected {expected}. "
            "Rebuild the cache using the exact results/split.csv row order."
        )
    return cache


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
