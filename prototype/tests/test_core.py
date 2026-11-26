"""Synthetic compatibility tests, not evaluations of the trained project models."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.io import MemoryFile
from sklearn.dummy import DummyClassifier

from core import PrototypeError, apply_haze, display_haze_pair, checkpoint_path, live_predict, read_multispectral_tiff
from piece_of_crop import BAND_NAMES, SmallCNN, build_resnet


def create_stats(root):
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"band": BAND_NAMES, "clip_low": [0]*13, "clip_high": [2000]*13,
                  "mean": [700]*13, "std": [300]*13}).to_csv(out/"band_stats.csv", index=False)


def create_raw():
    raw = np.random.default_rng(1).integers(300, 1600, size=(13, 64, 64), dtype=np.uint16)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=64, width=64, count=13, dtype="uint16") as dst:
            dst.write(raw)
        data = mem.read()
    return data


def test_tiff_reads_and_validates_shape():
    data = create_raw()
    assert read_multispectral_tiff(data).shape == (13, 64, 64)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=64, height=64, count=3, dtype="uint16") as ds:
            ds.write(np.ones((3,64,64),dtype="uint16"))
        with pytest.raises(PrototypeError, match="13-band"):
            read_multispectral_tiff(mem.read())


def test_cnn_checkpoint_and_preprocessing(tmp_path):
    create_stats(tmp_path)
    weights = checkpoint_path(tmp_path, "Small CNN", "13band", 0)
    weights.parent.mkdir(parents=True)
    model = SmallCNN(in_channels=13, num_classes=10)
    torch.save(model.state_dict(), weights)  # latest notebook 04 format
    raw = read_multispectral_tiff(create_raw())
    probs = live_predict(tmp_path, raw, "Small CNN", "13band", 0)
    assert probs.shape == (10,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_wrapped_cnn_checkpoint(tmp_path):
    create_stats(tmp_path)
    weights = checkpoint_path(tmp_path, "Small CNN", "rgb", 1)
    weights.parent.mkdir(parents=True)
    torch.save({"state_dict": SmallCNN(3).state_dict(), "input": "rgb"}, weights)
    probs = live_predict(tmp_path, read_multispectral_tiff(create_raw()), "Small CNN", "rgb", 1)
    assert probs.shape == (10,)


def test_random_forest_partial_classes(tmp_path):
    create_stats(tmp_path)
    raw = read_multispectral_tiff(create_raw())
    for input_type, nf in [("rgb", 6), ("13band", 28)]:
        cp = checkpoint_path(tmp_path, "Random Forest", input_type, 0)
        cp.parent.mkdir(parents=True, exist_ok=True)
        model = DummyClassifier(strategy="prior")
        model.fit(np.random.default_rng(1).random((4, nf)), [0, 0, 3, 3])
        joblib.dump(model, cp)
        p = live_predict(tmp_path, raw, "Random Forest", input_type, 0)
        assert p.shape == (10,)
        assert p.sum() == pytest.approx(1, abs=1e-5)
        assert np.count_nonzero(p) == 2


def test_resnet_loading_without_pretrained_download(tmp_path):
    create_stats(tmp_path)
    cp = checkpoint_path(tmp_path, "ResNet18", "rgb", 0)
    cp.parent.mkdir(parents=True)
    torch.save(build_resnet(in_channels=3, pretrained=False).state_dict(), cp)
    p = live_predict(tmp_path, read_multispectral_tiff(create_raw()), "ResNet18", "rgb", 0)
    assert len(p) == 10
    assert p.sum() == pytest.approx(1, abs=1e-5)


def test_missing_checkpoint_is_not_faked(tmp_path):
    create_stats(tmp_path)
    with pytest.raises(PrototypeError, match="missing"):
        live_predict(tmp_path, np.zeros((13,64,64),dtype="float32"), "ResNet18", "13band", 0)


def test_apply_haze_none_is_unchanged_and_haze_moves_towards_airlight(tmp_path):
    create_stats(tmp_path)
    raw = np.full((13, 64, 64), 500, dtype=np.float32)
    assert apply_haze(tmp_path, raw, 0.0, "wavelength") is raw
    hazy = apply_haze(tmp_path, raw, 0.2, "uniform")
    assert hazy.shape == raw.shape
    assert np.allclose(hazy, 0.8 * 500 + 0.2 * 2000)


def test_haze_preview_gets_brighter_and_flatter(tmp_path):
    create_stats(tmp_path)
    raw = np.random.default_rng(0).uniform(300, 1600, size=(13, 64, 64)).astype(np.float32)
    clear, hazy = display_haze_pair(tmp_path, raw, apply_haze(tmp_path, raw, 0.4, "uniform"))
    assert clear.shape == hazy.shape == (64, 64, 3)
    assert hazy.mean() > clear.mean() and hazy.std() < clear.std()
