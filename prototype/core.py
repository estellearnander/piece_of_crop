"""Live, single-patch inference using the project's original training modules.

This module never retrains a model and never downloads weights. Only checkpoint
files already stored in the project's trusted local models/ directory are read.
"""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PREFIXES = {"Random Forest": "rf", "Small CNN": "cnn", "ResNet18": "resnet"}
INPUT_CHOICES = {"RGB": "rgb", "13 bands": "13band"}
# Haze levels and profiles tested in notebook 07 (share of the signal from haze at 550 nm).
HAZE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.63]
HAZE_PROFILES = {"Wavelength (Ångström)": "wavelength", "Uniform (control)": "uniform"}


class PrototypeError(RuntimeError):
    """A problem with the local experiment inputs rather than a UI issue."""


def read_multispectral_tiff(source: str | Path | bytes) -> np.ndarray:
    """Read a full 13-band, 64x64 multispectral EuroSAT GeoTIFF as float32.

    Source can be a local file or uploaded bytes. A PNG/JPEG is not a valid
    substitute for multispectral satellite input.
    """
    import rasterio
    from rasterio.io import MemoryFile

    try:
        if isinstance(source, bytes):
            with MemoryFile(source) as mem:
                with mem.open() as ds:
                    raw = ds.read()
        else:
            with rasterio.open(Path(source)) as ds:
                raw = ds.read()
    except Exception as e:
        raise PrototypeError(f"Could not read this GeoTIFF: {e}") from e
    if raw.shape != (13, 64, 64):
        raise PrototypeError(
            f"Expected a 13-band EuroSAT patch of 64 × 64 pixels, got {tuple(raw.shape)}. "
            "Use the multispectral EuroSAT GeoTIFF, not the RGB dataset."
        )
    if not np.isfinite(raw).all():
        raise PrototypeError("The GeoTIFF contains NaN or infinite pixel values.")
    return raw.astype(np.float32)


def display_rgb(raw: np.ndarray) -> np.ndarray:
    from piece_of_crop.data import to_rgb_image
    return np.clip(to_rgb_image(raw), 0, 1)


def read_training_stats(root: Path) -> pd.DataFrame:
    from piece_of_crop.data import BAND_NAMES

    path = root / "results" / "band_stats.csv"
    if not path.is_file():
        raise PrototypeError("Missing results/band_stats.csv; run notebook 02 first.")
    table = pd.read_csv(path)
    required = {"band", "clip_low", "clip_high", "mean", "std"}
    if not required.issubset(table.columns):
        raise PrototypeError("band_stats.csv does not contain the expected training columns.")
    if table["band"].tolist() != list(BAND_NAMES):
        raise PrototypeError("band_stats.csv band order differs from src/piece_of_crop/data.py.")
    if (table["std"] <= 0).any() or (table["clip_low"] >= table["clip_high"]).any():
        raise PrototypeError("Invalid band_stats.csv normalisation values.")
    return table


def apply_haze(root: Path, raw: np.ndarray, strength: float, profile: str) -> np.ndarray:
    """Add the notebook 07 haze to raw 13-band values; strength 0 returns the patch unchanged.

    The airlight is each band's upper clip limit from the training statistics, as in notebook 07.
    """
    if strength == 0:
        return raw
    from piece_of_crop.haze import add_haze
    stats = read_training_stats(root)
    return add_haze(raw, strength, profile, stats["clip_high"].to_numpy()).astype(np.float32)


def checkpoint_path(root: Path, model_name: str, input_type: str, seed: int) -> Path:
    if model_name not in MODEL_PREFIXES or input_type not in ("rgb", "13band"):
        raise PrototypeError("Unrecognised model or input type.")
    if seed not in (0, 1, 2):
        raise PrototypeError("Seed must be 0, 1 or 2.")
    ext = ".joblib" if model_name == "Random Forest" else ".pt"
    return root / "models" / f"{MODEL_PREFIXES[model_name]}_{input_type}_seed{seed}{ext}"


@lru_cache(maxsize=4)
def _load_trusted_model(path_str: str, model_name: str, input_type: str, file_mtime_ns: int):
    """Cache only fixed, local model files. mtime invalidates replaced weights."""
    path = Path(path_str)
    if model_name == "Random Forest":
        import joblib
        return joblib.load(path)  # Only use trusted checkpoints created by the team.

    import torch
    from piece_of_crop import SmallCNN, build_resnet

    channels = 3 if input_type == "rgb" else 13
    if model_name == "Small CNN":
        model = SmallCNN(in_channels=channels, num_classes=10)
    else:
        # pretrained=False is essential: inference must not redownload ImageNet weights.
        model = build_resnet(in_channels=channels, num_classes=10, pretrained=False)

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch, trusted team checkpoints only.
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    if isinstance(model, SmallCNN):
        state = {
            ("classifier." + key[len("head.2."):])
            if key.startswith("head.2.") else key: value
            for key, value in state.items()
        }

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def live_predict(root: Path, raw: np.ndarray, model_name: str, input_type: str, seed: int) -> np.ndarray:
    """Return 10 class probabilities, applying the precise training transforms."""
    if raw.shape != (13, 64, 64):
        raise PrototypeError("Live inference requires raw 13-band 64 × 64 imagery.")
    path = checkpoint_path(root, model_name, input_type, seed)
    if not path.is_file():
        raise PrototypeError(f"Model checkpoint is missing: models/{path.name}")
    stats = read_training_stats(root)
    try:
        model = _load_trusted_model(str(path), model_name, input_type, path.stat().st_mtime_ns)
        if model_name == "Random Forest":
            from piece_of_crop.features import extract_rf_features
            rgb_features, ms_features = extract_rf_features(raw[None], stats)
            vector = rgb_features if input_type == "rgb" else ms_features
            partial = np.asarray(model.predict_proba(vector)[0], dtype=float)
            classes = np.asarray(model.classes_, dtype=int)
            probs = np.zeros(10, dtype=float)
            if (classes < 0).any() or (classes >= 10).any():
                raise PrototypeError("Random Forest checkpoint uses unexpected class IDs.")
            probs[classes] = partial
        else:
            import torch
            from torchvision.transforms import v2
            from piece_of_crop.data import normalise, select_bands
            # Same band clipping + training-only z-scoring as EuroSATDataset.
            image = normalise(raw, stats)
            image = np.ascontiguousarray(select_bands(image, input_type), dtype=np.float32)
            tensor = torch.from_numpy(image)
            if model_name == "ResNet18":
                # Same antialiased resize as make_loaders(..., image_size=224).
                tensor = v2.Resize((224, 224), antialias=True)(tensor)
            with torch.inference_mode():
                logits = model(tensor.unsqueeze(0))
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    except PrototypeError:
        raise
    except Exception as e:
        raise PrototypeError(f"Could not run {model_name} ({input_type}, seed {seed}): {e}") from e

    if probs.shape != (10,) or not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1.0, atol=1e-3):
        raise PrototypeError("The model did not return 10 valid class probabilities.")
    return probs


def saved_prediction(root: Path, sample_file: str, sample_label: int,
                     model_name: str, input_type: str, seed: int):
    """Retrieve a genuine held-out prediction; never present it as live inference."""
    prefix = MODEL_PREFIXES[model_name]
    path = root / "results" / "predictions" / f"{prefix}_{input_type}_seed{seed}.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    needed = {"file", "y_true", "y_pred", *[f"p_{i}" for i in range(10)]}
    if not needed.issubset(table.columns):
        return None
    matches = table[table["file"].eq(sample_file)]
    if len(matches) != 1 or int(matches.iloc[0]["y_true"]) != int(sample_label):
        return None
    row = matches.iloc[0]
    probs = row[[f"p_{i}" for i in range(10)]].to_numpy(dtype=float)
    if not np.isclose(probs.sum(), 1.0, atol=1e-3) or int(probs.argmax()) != int(row["y_pred"]):
        return None
    return probs


def check_local_resources(root: Path) -> dict:
    """Small resource audit; no data or weights are loaded."""
    return {
        "split": (root / "results" / "split.csv").is_file(),
        "band_stats": (root / "results" / "band_stats.csv").is_file(),
        "test_predictions": len(list((root / "results" / "predictions").glob("*.csv"))),
        "checkpoints": len(list((root / "models").glob("*.pt"))) + len(list((root / "models").glob("*.joblib"))),
        "raw_dataset": (root / "data" / "EuroSAT_MS").is_dir(),
    }
