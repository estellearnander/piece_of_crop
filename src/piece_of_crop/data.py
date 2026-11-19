"""Loading, checking and summarising EuroSAT multispectral patches (owner: Estelle).

Reading GeoTIFFs is not covered in the course, so it follows the rasterio
documentation (external): rasterio.open(), dataset.read() returns an array
ordered (bands, rows, columns), and dataset.crs / dataset.bounds give the
georeferencing ("Reading Datasets" and "Dataset georeferencing" in the docs).
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import BallTree


# Class folders, in alphabetical order. The label of a class is its position here.
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Band order inside the EuroSAT_MS GeoTIFFs. B8A is the LAST band, not the 9th.
# Checked in 01_data_exploration: band index 9 has a mean of about 12, which only
# fits B10 (cirrus), so B09, B10, B11 and B12 come before B8A.
BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A",
]

# Positions of red (B04), green (B03) and blue (B02) in BAND_NAMES.
RGB_BAND_INDICES = [3, 2, 1]

EXPECTED_SHAPE = (13, 64, 64)


def list_files(data_dir: str | Path) -> pd.DataFrame:
    """Return one row per .tif file with its path, class name and label."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    rows = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        for path in sorted(class_dir.glob("*.tif")):
            rows.append({"file": path.name, "path": str(path),
                         "class_name": class_name, "label": label})
    return pd.DataFrame(rows)


def read_bands(path: str | Path) -> np.ndarray:
    """Return the raw band values of one patch, shape (13, 64, 64), as float32.

    No normalisation happens here, so later steps (normalisation, haze) can work
    on the raw values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with rasterio.open(path) as dataset:
        bands = dataset.read()
    return bands.astype(np.float32)


def read_center_lonlat(path: str | Path) -> tuple[float, float]:
    """Return the longitude and latitude of the centre of one patch."""
    with rasterio.open(path) as dataset:
        bounds = dataset.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        lons, lats = transform(dataset.crs, "EPSG:4326", [center_x], [center_y])
    return lons[0], lats[0]


def check_file(path: str | Path) -> str:
    """Return "ok", or a short description of what is wrong with the file."""
    try:
        bands = read_bands(path)
    except Exception as error:  # a broken file can raise many kinds of error
        return f"cannot be read: {error}"
    if bands.shape != EXPECTED_SHAPE:
        return f"wrong shape {bands.shape}"
    if not np.isfinite(bands).all():
        return "contains NaN or infinite values"
    if bands.max() == 0:
        return "all values are zero"
    return "ok"


def pixel_fingerprint(bands: np.ndarray) -> str:
    """Return an MD5 hash of the pixel values, to find exact duplicate images."""
    return hashlib.md5(bands.tobytes()).hexdigest()


def summarise_files(files: pd.DataFrame) -> pd.DataFrame:
    """Read every file once and return one row of facts per file.

    Columns added: status, fingerprint, lon, lat, and for each band its mean,
    standard deviation, minimum and maximum over the 64 x 64 pixels.
    Each file is opened only once, because reading 27,000 files is the slow part.
    """
    rows = []
    for path in files["path"]:
        row = {"path": path}
        try:
            with rasterio.open(path) as dataset:
                bands = dataset.read().astype(np.float32)
                bounds = dataset.bounds
                crs = dataset.crs
        except Exception as error:  # a broken file can raise many kinds of error
            row["status"] = f"cannot be read: {error}"
            rows.append(row)
            continue

        if bands.shape != EXPECTED_SHAPE:
            row["status"] = f"wrong shape {bands.shape}"
        elif not np.isfinite(bands).all():
            row["status"] = "contains NaN or infinite values"
        elif bands.max() == 0:
            row["status"] = "all values are zero"
        else:
            row["status"] = "ok"

        if row["status"] == "ok":
            row["fingerprint"] = pixel_fingerprint(bands)
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            lons, lats = transform(crs, "EPSG:4326", [center_x], [center_y])
            row["lon"], row["lat"] = lons[0], lats[0]
            flat = bands.reshape(13, -1)
            for i, band in enumerate(BAND_NAMES):
                row[f"mean_{band}"] = flat[i].mean()
                row[f"std_{band}"] = flat[i].std()
                row[f"min_{band}"] = flat[i].min()
                row[f"max_{band}"] = flat[i].max()
        rows.append(row)
    return files.merge(pd.DataFrame(rows), on="path")


def find_duplicates(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the files whose pixels are identical to at least one other file."""
    duplicated = summary["fingerprint"].duplicated(keep=False)
    return summary.loc[duplicated & summary["fingerprint"].notna()].sort_values("fingerprint")


def to_rgb_image(bands: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Return a (64, 64, 3) image in [0, 1] for display only.

    Bands 4, 3 and 2 are each stretched between their own 2nd and 98th percentile,
    so the patch is visible and no single band dominates the colour. This is for
    plots only: the models get the raw bands.
    """
    rgb = bands[RGB_BAND_INDICES].transpose(1, 2, 0)
    image = np.zeros_like(rgb)
    for channel in range(3):
        values = rgb[:, :, channel]
        lo = np.percentile(values, low)
        hi = np.percentile(values, high)
        image[:, :, channel] = np.clip((values - lo) / (hi - lo + 1e-8), 0, 1)
    return image


# ---------------------------------------------------------------------------
# Splitting (notebook 02)
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0


def assign_blocks(table: pd.DataFrame, block_size_deg: float) -> pd.Series:
    """Return the grid cell ("block") of each patch, as a text id like "12_101".

    The grid has square cells of block_size_deg degrees of longitude and latitude.
    """
    column = np.floor(table["lon"] / block_size_deg).astype(int)
    row = np.floor(table["lat"] / block_size_deg).astype(int)
    return column.astype(str) + "_" + row.astype(str)


def nearest_distance_km(from_table: pd.DataFrame, to_table: pd.DataFrame) -> np.ndarray:
    """For each patch in from_table, return the distance in km to the nearest patch in to_table."""
    tree = BallTree(np.radians(to_table[["lat", "lon"]].to_numpy()), metric="haversine")
    distances, _ = tree.query(np.radians(from_table[["lat", "lon"]].to_numpy()), k=1)
    return distances[:, 0] * EARTH_RADIUS_KM


def spatial_split(table: pd.DataFrame, block_size_deg: float = 0.5,
                  n_folds: int = 7, seed: int = 0) -> pd.DataFrame:
    """Split patches into train, val and test so that a block is never in two splits.

    StratifiedGroupKFold (scikit-learn) makes n_folds folds that keep whole blocks
    together and keep the class proportions as close as possible. Fold 0 becomes
    the test set, fold 1 the validation set and the rest the training set.
    """
    result = table.copy()
    result["block"] = assign_blocks(result, block_size_deg)

    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_row = np.zeros(len(result), dtype=int)
    for fold_number, (_, fold_rows) in enumerate(folds.split(result, result["label"], result["block"])):
        fold_of_row[fold_rows] = fold_number

    result["split"] = "train"
    result.loc[fold_of_row == 0, "split"] = "test"
    result.loc[fold_of_row == 1, "split"] = "val"
    return result


def apply_buffer(split_table: pd.DataFrame, buffer_km: float = 1.0) -> pd.DataFrame:
    """Mark patches that sit too close to a held-out patch as "buffer" (not used).

    Training patches closer than buffer_km to a val or test patch, and val patches
    closer than buffer_km to a test patch, are marked. This removes the neighbours
    that sit on the edge between two blocks.
    """
    result = split_table.copy()

    train = result[result["split"] == "train"]
    held_out = result[result["split"].isin(["val", "test"])]
    if len(train) > 0 and len(held_out) > 0:
        too_close = train.index[nearest_distance_km(train, held_out) < buffer_km]
        result.loc[too_close, "split"] = "buffer"

    val = result[result["split"] == "val"]
    test = result[result["split"] == "test"]
    if len(val) > 0 and len(test) > 0:
        too_close = val.index[nearest_distance_km(val, test) < buffer_km]
        result.loc[too_close, "split"] = "buffer"
    return result


# ---------------------------------------------------------------------------
# Normalisation (notebook 02)
# ---------------------------------------------------------------------------

def sample_pixels(paths, pixels_per_file: int = 200, seed: int = 0) -> np.ndarray:
    """Return a random sample of pixels from the given files, shape (n_pixels, 13)."""
    rng = np.random.default_rng(seed)
    samples = []
    for path in paths:
        flat = read_bands(path).reshape(13, -1)
        chosen = rng.choice(flat.shape[1], size=pixels_per_file, replace=False)
        samples.append(flat[:, chosen].T)
    return np.concatenate(samples, axis=0)


def compute_band_stats(train_paths, low_percentile: float = 0.1, high_percentile: float = 99.9,
                       pixels_per_file: int = 200, seed: int = 0) -> pd.DataFrame:
    """Return per-band clip limits, mean and standard deviation from training files only.

    Step 1: the clip limits are percentiles of a random pixel sample.
    Step 2: the mean and standard deviation are computed over ALL training pixels
    after clipping, reading each file once more.
    """
    pixels = sample_pixels(train_paths, pixels_per_file, seed)
    clip_low = np.percentile(pixels, low_percentile, axis=0)
    clip_high = np.percentile(pixels, high_percentile, axis=0)

    total = np.zeros(13)
    total_squared = np.zeros(13)
    count = 0
    for path in train_paths:
        flat = read_bands(path).reshape(13, -1).astype(np.float64)
        clipped = np.clip(flat, clip_low[:, None], clip_high[:, None])
        total += clipped.sum(axis=1)
        total_squared += (clipped ** 2).sum(axis=1)
        count += clipped.shape[1]
    mean = total / count
    std = np.sqrt(total_squared / count - mean ** 2)

    return pd.DataFrame({
        "band": BAND_NAMES,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "mean": mean,
        "std": std,
    })


def normalise(bands: np.ndarray, stats: pd.DataFrame) -> np.ndarray:
    """Clip each band to its limits and scale it to mean 0 and standard deviation 1.

    bands has shape (13, H, W) with the raw values. The statistics come from the
    training split only (see compute_band_stats).
    """
    low = stats["clip_low"].to_numpy(dtype=np.float32)[:, None, None]
    high = stats["clip_high"].to_numpy(dtype=np.float32)[:, None, None]
    mean = stats["mean"].to_numpy(dtype=np.float32)[:, None, None]
    std = stats["std"].to_numpy(dtype=np.float32)[:, None, None]
    clipped = np.clip(bands, low, high)
    return ((clipped - mean) / std).astype(np.float32)


def select_bands(bands: np.ndarray, input_type: str) -> np.ndarray:
    """Return all 13 bands ("13band") or bands 4, 3, 2 in that order ("rgb")."""
    if input_type == "13band":
        return bands
    if input_type == "rgb":
        return bands[RGB_BAND_INDICES]
    raise ValueError(f"input_type must be 'rgb' or '13band', not {input_type!r}")
