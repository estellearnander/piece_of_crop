import numpy as np
import pandas as pd
import pytest
import torch

from piece_of_crop.cnn import SmallCNN, build_cnn
from piece_of_crop.train import predictions_table, run_experiment, set_seed


def _toy_inputs(n_per_split=8):
    rng = np.random.default_rng(0)
    n = 3 * n_per_split
    cache = rng.integers(0, 5000, (n, 13, 64, 64)).astype(np.uint16)
    split = pd.DataFrame({
        "file": [f"patch_{i}.tif" for i in range(n)],
        "cache_row": np.arange(n),
        "label": np.arange(n) % 10,
        "split": ["train"] * n_per_split + ["val"] * n_per_split + ["test"] * n_per_split,
    })
    stats = pd.DataFrame({"clip_low": [0.0] * 13, "clip_high": [5000.0] * 13,
                          "mean": [2500.0] * 13, "std": [1000.0] * 13})
    return split, cache, stats


@pytest.mark.parametrize("input_type, channels", [("rgb", 3), ("13band", 13)])
def test_cnn_gives_10_scores_per_patch(input_type, channels):
    model = build_cnn(input_type)
    out = model(torch.zeros(4, channels, 64, 64))
    assert out.shape == (4, 10)


def test_rgb_and_13band_cnn_only_differ_in_first_layer():
    rgb, ms = SmallCNN(3), SmallCNN(13)
    rgb_shapes = [p.shape for p in rgb.parameters()]
    ms_shapes = [p.shape for p in ms.parameters()]
    assert rgb_shapes[1:] == ms_shapes[1:]
    assert rgb_shapes[0][1] == 3 and ms_shapes[0][1] == 13


def test_same_seed_gives_same_starting_weights():
    set_seed(0)
    a = build_cnn("rgb").state_dict()
    set_seed(0)
    b = build_cnn("rgb").state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_predictions_table_has_the_shared_columns():
    split = pd.DataFrame({"file": ["a.tif", "b.tif"], "label": [3, 7]})
    probs = np.full((2, 10), 0.05)
    probs[0, 3], probs[1, 1] = 0.55, 0.55
    table = predictions_table(split, probs)
    assert list(table.columns) == ["file", "true_label", "pred_label"] + [f"prob_{k}" for k in range(10)]
    assert list(table["pred_label"]) == [3, 1]


def test_run_experiment_saves_predictions_curves_and_weights(tmp_path):
    split, cache, stats = _toy_inputs()
    row = run_experiment("cnn", build_cnn, split, cache, stats, "13band", seed=0,
                         project_root=tmp_path, batch_size=4, max_epochs=2,
                         device=torch.device("cpu"), verbose=False)
    preds = pd.read_csv(tmp_path / "results" / "predictions" / "cnn_13band_seed0.csv")
    assert len(preds) == (split["split"] == "test").sum()
    assert list(preds["file"]) == list(split.loc[split["split"] == "test", "file"])
    assert np.allclose(preds[[f"prob_{k}" for k in range(10)]].sum(axis=1), 1, atol=1e-4)
    assert (tmp_path / "results" / "training" / "cnn_13band_seed0.csv").exists()
    assert (tmp_path / "results" / "figures" / "curves_cnn_13band_seed0.png").exists()
    assert (tmp_path / "models" / "cnn_13band_seed0.pt").exists()
    assert 1 <= row["best_epoch"] <= row["epochs_run"] <= 2
    assert 0 <= row["test_accuracy"] <= 1
