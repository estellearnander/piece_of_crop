"""Simulated haze on the raw bands of the test patches (notebook 07). No model is retrained.

Haze model (external: He, Sun and Tang, 2011, equation 1): a hazy value is the clear
value seen through the haze plus light scattered by the haze itself,
    hazy = t * clear + (1 - t) * airlight,
where t is the share of light that passes the haze. Here t = exp(-2 * tau): the light
crosses the haze twice, from the sun to the ground and from the ground to the satellite,
and exp(-tau) is the transmission of one vertical pass (Beer-Lambert). tau is the
aerosol optical depth of a band. With the "wavelength" profile it follows the Angstrom
law, tau = aod_550 * (wavelength / 550 nm) ** -1.3, so short wavelengths get more haze;
with the "uniform" profile every band gets the tau of 550 nm.
"""

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .data import EuroSATDataset
from .features import extract_rf_features

# Central wavelength of each band in nm, in the band order of the files (B01 ... B12, B8A)
# (external: USGS EROS Archive, Sentinel-2, table "Spectral Bands and Resolution").
WAVELENGTHS_NM = np.array([443, 490, 560, 665, 705, 740, 783, 842, 945, 1375, 1610, 2190, 865], dtype=np.float32)
ANGSTROM_EXPONENT = 1.3  # average for natural atmospheres (external, see notebook 07)
REFERENCE_NM = 550.0  # aerosol optical depth is reported at 550 nm


def haze_share(aod: float, profile: str) -> np.ndarray:
    """Share of each band's value that comes from the haze, (1 - t), for 13 bands."""
    if profile == "wavelength":
        tau = aod * (WAVELENGTHS_NM / REFERENCE_NM) ** -ANGSTROM_EXPONENT
    elif profile == "uniform":
        tau = np.full(13, aod, dtype=np.float32)
    else:
        raise ValueError(f"Unknown haze profile {profile}")
    return (1 - np.exp(-2 * tau)).astype(np.float32)


def add_haze(raw, aod: float, profile: str, airlight) -> np.ndarray:
    """Add haze to raw bands of shape (13, H, W) or (N, 13, H, W)."""
    raw = np.asarray(raw, dtype=np.float32)
    share = haze_share(aod, profile).reshape(13, 1, 1)
    airlight = np.asarray(airlight, dtype=np.float32).reshape(13, 1, 1)
    return (1 - share) * raw + share * airlight


def hazy_test_loader(test_rows: pd.DataFrame, cache, band_stats: pd.DataFrame, input_type: str,
                     aod: float, profile: str, airlight, image_size=None, batch_size: int = 32) -> DataLoader:
    """Test loader that adds haze to the raw bands and then preprocesses them like the clean data."""
    resize = v2.Resize((image_size, image_size), antialias=True) if image_size is not None else None
    dataset = EuroSATDataset(test_rows, cache, band_stats, input_type, transform=resize,
                             raw_transform=lambda raw: add_haze(raw, aod, profile, airlight))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def rf_hazy_probabilities(model, test_rows: pd.DataFrame, cache, band_stats: pd.DataFrame, input_type: str,
                          aod: float, profile: str, airlight, batch_size: int = 500) -> np.ndarray:
    """Class probabilities of a saved random forest on the hazy test patches."""
    probabilities = []
    for start in range(0, len(test_rows), batch_size):
        rows = test_rows["cache_row"].to_numpy()[start:start + batch_size]
        hazy = add_haze(cache[rows], aod, profile, airlight)
        X_rgb, X_ms = extract_rf_features(hazy, band_stats)
        probabilities.append(model.predict_proba(X_rgb if input_type == "rgb" else X_ms))
    return np.vstack(probabilities)
