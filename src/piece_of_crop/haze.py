"""Simulated haze on the raw bands of the test patches (notebook 07). No model is retrained.

Haze model (external: He, Sun and Tang, 2011, equation 1): a hazy value is the clear
value seen through the haze plus light scattered by the haze itself,
    hazy = (1 - share) * clear + share * airlight.
The strength of the haze is the share at 550 nm. With the "wavelength" profile the other
bands follow the Angstrom law: the optical depth of a band is tau_550 * (wavelength / 550 nm)
** -1.3, and share = 1 - exp(-2 * tau) (two passes through the haze, Beer-Lambert), so short
wavelengths get more haze. With the "uniform" profile every band gets the share of 550 nm.
The strength is not calibrated to a real aerosol optical depth (see notebook 07, section 2).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .cnn import SmallCNN
from .data import CLASS_NAMES, EuroSATDataset
from .evaluate import MODEL_FILES
from .features import extract_rf_features
from .resnet import IMAGE_SIZE, build_resnet
from .train import predict

# Central wavelength of each band in nm, in the band order of the files (B01 ... B12, B8A)
# (external: USGS EROS Archive, Sentinel-2, table "Spectral Bands and Resolution").
WAVELENGTHS_NM = np.array([443, 490, 560, 665, 705, 740, 783, 842, 945, 1375, 1610, 2190, 865], dtype=np.float32)
ANGSTROM_EXPONENT = 1.3  # average for natural atmospheres (external, see notebook 07)
REFERENCE_NM = 550.0  # the strength of the haze is given at 550 nm


def haze_share(strength: float, profile: str) -> np.ndarray:
    """Share of each band's value that comes from the haze, for 13 bands.

    strength: the share at 550 nm, between 0 (no haze) and 1 (only haze).
    """
    if profile == "uniform":
        return np.full(13, strength, dtype=np.float32)
    if profile != "wavelength":
        raise ValueError(f"Unknown haze profile {profile}")
    tau_550 = -np.log(1 - strength) / 2  # optical depth that gives this share at 550 nm
    tau = tau_550 * (WAVELENGTHS_NM / REFERENCE_NM) ** -ANGSTROM_EXPONENT
    return (1 - np.exp(-2 * tau)).astype(np.float32)


def add_haze(raw, strength: float, profile: str, airlight) -> np.ndarray:
    """Add haze to raw bands of shape (13, H, W) or (N, 13, H, W)."""
    raw = np.asarray(raw, dtype=np.float32)
    share = haze_share(strength, profile).reshape(13, 1, 1)
    airlight = np.asarray(airlight, dtype=np.float32).reshape(13, 1, 1)
    return (1 - share) * raw + share * airlight


def hazy_test_loader(test_rows: pd.DataFrame, cache, band_stats: pd.DataFrame, input_type: str,
                     strength: float, profile: str, airlight, image_size=None, batch_size: int = 32) -> DataLoader:
    """Test loader that adds haze to the raw bands and then preprocesses them like the clean data."""
    resize = v2.Resize((image_size, image_size), antialias=True) if image_size is not None else None
    dataset = EuroSATDataset(test_rows, cache, band_stats, input_type, transform=resize,
                             raw_transform=lambda raw: add_haze(raw, strength, profile, airlight))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def rf_hazy_probabilities(model, test_rows: pd.DataFrame, cache, band_stats: pd.DataFrame, input_type: str,
                          strength: float, profile: str, airlight, batch_size: int = 500) -> np.ndarray:
    """Class probabilities of a saved random forest on the hazy test patches."""
    probabilities = []
    for start in range(0, len(test_rows), batch_size):
        rows = test_rows["cache_row"].to_numpy()[start:start + batch_size]
        hazy = add_haze(cache[rows], strength, profile, airlight)
        X_rgb, X_ms = extract_rf_features(hazy, band_stats)
        probabilities.append(model.predict_proba(X_rgb if input_type == "rgb" else X_ms))
    return np.vstack(probabilities)


def load_saved_model(models_dir, model_name: str, input_type: str, seed: int, device="cpu"):
    """Load one of the 18 saved models (notebooks 03 to 05)."""
    path = Path(models_dir) / f"{MODEL_FILES[model_name]}_{input_type}_seed{seed}"
    if model_name == "RandomForest":
        forest = joblib.load(path.with_suffix(".joblib"))
        assert list(forest.classes_) == list(range(len(CLASS_NAMES)))  # probability columns are classes 0 to 9
        return forest
    channels = 3 if input_type == "rgb" else 13
    if model_name == "SmallCNN":
        net = SmallCNN(in_channels=channels, num_classes=len(CLASS_NAMES))
    else:
        net = build_resnet(in_channels=channels, num_classes=len(CLASS_NAMES), pretrained=False)
    net.load_state_dict(torch.load(path.with_suffix(".pt"), map_location="cpu"))
    return net.to(device)


def hazy_probabilities(model_name: str, model, test_rows: pd.DataFrame, cache, band_stats: pd.DataFrame,
                       input_type: str, strength: float, profile: str, airlight, device="cpu") -> np.ndarray:
    """Class probabilities of a saved model on the hazy test patches, in split order."""
    if model_name == "RandomForest":
        return rf_hazy_probabilities(model, test_rows, cache, band_stats, input_type, strength, profile, airlight)
    image_size = IMAGE_SIZE if model_name == "ResNet18" else None
    loader = hazy_test_loader(test_rows, cache, band_stats, input_type, strength, profile, airlight, image_size)
    labels, proba = predict(model, loader, device)
    assert np.array_equal(labels, test_rows["label"].to_numpy())  # same patches in the same order
    return proba
