from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from piece_of_crop import (
    SmallCNN,
    build_cnn,
    build_resnet,
    check_test_order,
    evaluate_model,
    fit,
    make_loaders,
    predictions_table,
    resize_images,
    set_seed,
)


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
    assert model(torch.zeros(4, channels, 64, 64)).shape == (4, 10)


def test_rgb_and_13band_cnn_only_differ_in_first_layer():
    rgb_shapes = [p.shape for p in SmallCNN(3).parameters()]
    ms_shapes = [p.shape for p in SmallCNN(13).parameters()]
    assert rgb_shapes[1:] == ms_shapes[1:]
    assert rgb_shapes[0][1] == 3 and ms_shapes[0][1] == 13


def test_same_seed_gives_same_starting_weights():
    set_seed(0)
    a = build_cnn("rgb").state_dict()
    set_seed(0)
    b = build_cnn("rgb").state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("input_type, channels", [("rgb", 3), ("13band", 13)])
def test_resnet_shapes_and_trainable_layers(input_type, channels):
    net = build_resnet(input_type, pretrained=False)
    assert net.conv1.in_channels == channels
    with torch.no_grad():
        assert net(resize_images(torch.zeros(2, channels, 64, 64), 128)).shape == (2, 10)
    trainable = {name.split(".")[0] for name, p in net.named_parameters() if p.requires_grad}
    expected = {"layer4", "fc"} | ({"conv1"} if input_type == "13band" else set())
    assert trainable == expected


def test_predictions_table_has_the_team_columns():
    probs = np.full((2, 10), 0.05)
    probs[0, 3], probs[1, 1] = 0.55, 0.55
    table = predictions_table(["a.tif", "b.tif"], [3, 7], probs)
    assert list(table.columns) == ["file", "y_true", "y_pred", "confidence"] + [f"p_{k}" for k in range(10)]
    assert list(table["y_pred"]) == [3, 1]
    assert np.allclose(table["confidence"], 0.55)


def test_fit_restores_a_checkpoint_and_test_order_matches_split():
    split, cache, stats = _toy_inputs()
    set_seed(0)
    loaders = make_loaders(split, cache, stats, "13band", batch_size=4, seed=0)
    model = build_cnn("13band")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history, loss_fn = fit(model, loaders, optimizer, torch.device("cpu"), max_epochs=2, patience=1,
                           run_info={"model": "SmallCNN", "input": "13band", "seed": 0})
    assert 1 <= len(history) <= 2
    outcome = evaluate_model(model, loaders["test"], loss_fn, torch.device("cpu"))
    check_test_order(split, outcome["y_true"])
    assert np.allclose(outcome["proba"].sum(axis=1), 1, atol=1e-5)
