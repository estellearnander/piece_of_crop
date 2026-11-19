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
