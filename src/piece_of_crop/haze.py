"""Synthetic haze on raw bands and prediction of saved models on hazy test
patches. Used by notebook 07. No model is retrained.

The haze is a heuristic stress test, not a physical atmosphere model:
x_hazy = (1 - a) * x + a * bright_reference, per band, applied to the raw
values before normalisation.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .cnn import SmallCNN
from .data import CLASS_NAMES, normalise, select_bands
from .features import extract_rf_features
from .resnet import build_resnet

# Band order B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A.
# Hand-set factors: short (blue) wavelengths get more haze than infrared.
WAVELENGTH_FACTORS = np.array([1.25, 1.20, 1.12, 1.00, 0.88, 0.80, 0.75, 0.68, 0.58, 0.58, 0.52, 0.48, 0.70],
                              dtype=np.float32)


def simulate_haze(raw, strength: float, profile: str, bright_reference) -> np.ndarray:
    """Mix raw bands with a bright reference value per band.

    raw: one patch (13, H, W) or a batch (N, 13, H, W) of raw values.
    strength: haze strength a between 0 (no haze) and 1.
    profile: "uniform" (same a for all bands) or "wavelength" (a times WAVELENGTH_FACTORS).
    bright_reference: 13 values, the training 99.9th percentile (clip_high in band_stats.csv).
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-3] != 13:
        raise ValueError("Expected original 13-band EuroSAT patches")
    if not 0 <= strength <= 1:
        raise ValueError("Haze strength must lie in [0, 1]")
    if profile == "uniform":
        factors = np.ones(13, dtype=np.float32)
    elif profile == "wavelength":
        factors = WAVELENGTH_FACTORS
    else:
        raise ValueError("Unknown haze profile")
    alpha = np.clip(strength * factors, 0, 1)
    leading = (1,) * (raw.ndim - 3)
    alpha = alpha.reshape(*leading, 13, 1, 1)
    air = np.asarray(bright_reference, dtype=np.float32).reshape(*leading, 13, 1, 1)
    return (1 - alpha) * raw + alpha * air


class HazyTestDataset(Dataset):
    """Test patches with haze added to the raw bands, then normalised like the clean data."""

    def __init__(self, rows: pd.DataFrame, cache, band_stats: pd.DataFrame, input_type: str,
                 profile: str, strength: float):
        self.rows = rows
        self.cache = cache
        self.band_stats = band_stats
        self.input_type = input_type
        self.profile = profile
        self.strength = strength
        self.bright_reference = band_stats["clip_high"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        raw = np.asarray(self.cache[int(row["cache_row"])], dtype=np.float32)
        hazy = simulate_haze(raw, self.strength, self.profile, self.bright_reference)
        x = select_bands(normalise(hazy, self.band_stats), self.input_type)
        return torch.from_numpy(np.ascontiguousarray(x)), int(row["label"])


def load_deep_model(models_dir, family: str, input_type: str, seed: int, device):
    """Load a saved CNN ("cnn") or ResNet ("resnet") for inference.

    Returns (model in evaluation mode, input size the model was trained on).
    """
    path = models_dir / f"{family}_{input_type}_seed{seed}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if family == "cnn":
        net = SmallCNN(in_channels=3 if input_type == "rgb" else 13, num_classes=len(CLASS_NAMES))
        net.load_state_dict(checkpoint)
        image_size = 64
    elif family == "resnet":
        net = build_resnet(input_type, pretrained=False, num_classes=len(CLASS_NAMES))
        net.load_state_dict(checkpoint["state_dict"])
        image_size = int(checkpoint["input_size"])
    else:
        raise ValueError(f"Unsupported model family {family}")
    return net.to(device).eval(), image_size


@torch.inference_mode()
def predict_deep(net, image_size, test_rows, cache, band_stats, input_type, profile, strength,
                 device, batch_size=32) -> np.ndarray:
    """Class probabilities of a CNN or ResNet on the hazy test patches."""
    dataset = HazyTestDataset(test_rows, cache, band_stats, input_type, profile, strength)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_y, all_prob = [], []
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        if image_size != 64:
            x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        all_prob.append(torch.softmax(net(x), dim=1).cpu().numpy())
        all_y.append(y.numpy())
    if not np.array_equal(np.concatenate(all_y), test_rows["label"].to_numpy(dtype=int)):
        raise ValueError("Hazy test patches are not in the order of the test rows.")
    return np.concatenate(all_prob)


def predict_rf(model, test_rows, cache, band_stats, input_type, profile, strength,
               batch_size=32) -> np.ndarray:
    """Class probabilities of a random forest on the hazy test patches.

    The features are computed with the same extract_rf_features as in notebook 03.
    """
    bright_reference = band_stats["clip_high"].to_numpy(dtype=np.float32)
    prob_list = []
    for start in range(0, len(test_rows), batch_size):
        part = test_rows.iloc[start:start + batch_size]
        raw = np.asarray(cache[part["cache_row"].to_numpy(dtype=int)], dtype=np.float32)
        hazy = simulate_haze(raw, strength, profile, bright_reference)
        with contextlib.redirect_stdout(io.StringIO()):
            X_rgb, X_ms = extract_rf_features(hazy, band_stats, batch_size=batch_size)
        X = X_rgb if input_type == "rgb" else X_ms
        raw_prob = model.predict_proba(X)
        proba = np.zeros((len(part), len(CLASS_NAMES)))
        for j, label in enumerate(model.classes_):
            proba[:, int(label)] = raw_prob[:, j]
        prob_list.append(proba)
    return np.vstack(prob_list)
