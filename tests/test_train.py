import numpy as np
import pandas as pd
import pytest
import torch

from piece_of_crop import (SmallCNN, build_resnet, make_loaders, predict, predictions_table, seed_all,
                           train_model, tune_random_forest)


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


@pytest.mark.parametrize("channels", [3, 13])
def test_cnn_gives_10_logits_per_patch(channels):
    assert SmallCNN(in_channels=channels)(torch.zeros(4, channels, 64, 64)).shape == (4, 10)


def test_rgb_and_13band_cnn_only_differ_in_first_layer():
    rgb_shapes = [p.shape for p in SmallCNN(3).parameters()]
    ms_shapes = [p.shape for p in SmallCNN(13).parameters()]
    assert rgb_shapes[1:] == ms_shapes[1:]
    assert rgb_shapes[0][1] == 3 and ms_shapes[0][1] == 13


def test_same_seed_gives_same_starting_weights():
    seed_all(0)
    a = SmallCNN(3).state_dict()
    seed_all(0)
    b = SmallCNN(3).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("input_type, channels", [("rgb", 3), ("13band", 13)])
def test_resnet_shapes_and_trainable_layers(input_type, channels):
    net = build_resnet(input_type, pretrained=False)
    assert net.conv1.in_channels == channels
    with torch.no_grad():
        assert net(torch.zeros(2, channels, 128, 128)).shape == (2, 10)
    trainable = {name.split(".")[0] for name, p in net.named_parameters() if p.requires_grad}
    assert trainable == {"layer4", "fc"} | ({"conv1"} if input_type == "13band" else set())


def test_training_keeps_the_best_epoch_and_predicts_in_split_order():
    split, cache, stats = _toy_inputs()
    seed_all(0)
    loaders = make_loaders(split, cache, stats, "13band", batch_size=4)
    model = SmallCNN(in_channels=13)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history, best_epoch = train_model(model, loaders["train"], loaders["val"], optimizer, 2, torch.device("cpu"))
    assert len(history["val_loss"]) == 2
    assert history["val_loss"][best_epoch - 1] == min(history["val_loss"])
    y_true, proba = predict(model, loaders["test"], torch.device("cpu"))
    assert y_true.tolist() == split.loc[split["split"] == "test", "label"].tolist()
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)


def test_predictions_table_has_the_team_columns():
    probs = np.full((2, 10), 0.05)
    probs[0, 3], probs[1, 1] = 0.55, 0.55
    table = predictions_table(["a.tif", "b.tif"], [3, 7], probs)
    assert list(table.columns) == ["file", "y_true", "y_pred", "confidence"] + [f"p_{k}" for k in range(10)]
    assert list(table["y_pred"]) == [3, 1]


def test_tuning_picks_the_best_validation_combination():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)
    table, best = tune_random_forest(X[:150], y[:150], X[150:], y[150:],
                                     {"max_depth": [1, None], "min_samples_leaf": [1, 5]}, n_estimators=20)
    assert len(table) == 4
    assert table["val_macro_f1"].is_monotonic_decreasing
    assert best == table.loc[0, "params"]
