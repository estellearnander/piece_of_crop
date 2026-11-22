"""Training and validation loops shared by the CNN and the ResNet (owner: Rishi).

loraine - you can use run_experiment for the resnet too, just pass your own
build function instead of build_cnn

settings (same for every model unless we change it): cross entropy, adam,
lr 1e-3, batch 128, max 30 epochs, early stopping on val loss (patience 5)
"""

import copy
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .data import make_loaders

NUM_CLASSES = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    # cuda if you have it, mac gpu (mps) otherwise, else cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    # one pass through the data. trains if you give it an optimizer, otherwise just evaluates
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            loss = loss_fn(scores, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (scores.argmax(1) == labels).sum().item()
            seen += len(labels)
    return total_loss / seen, correct / seen


def fit(model, train_loader, val_loader, device, max_epochs: int = 30, lr: float = 1e-3,
        weight_decay: float = 0.0, patience: int = 5, verbose: bool = True):
    # trains until val loss stops improving, then loads back the best epoch
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    history = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc,
                        "seconds": time.time() - start})
        if verbose:
            print(f"  epoch {epoch:2d}  train loss {train_loss:.3f} acc {train_acc:.3f}  "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f}  ({time.time() - start:.0f}s)")

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"  stopping early, best epoch was {best_epoch}")
            break

    model.load_state_dict(best_state)
    return pd.DataFrame(history), best_epoch


@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    model.to(device).eval()
    probs = [torch.softmax(model(images.to(device)), dim=1).cpu() for images, _ in loader]
    return torch.cat(probs).numpy()


def predictions_table(split_rows: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    # the format we agreed on: file, true_label, pred_label, prob_0..prob_9
    # (works because val/test loaders dont shuffle so the order matches)
    table = pd.DataFrame({
        "file": split_rows["file"].to_numpy(),
        "true_label": split_rows["label"].to_numpy(),
        "pred_label": probs.argmax(1),
    })
    for k in range(probs.shape[1]):
        table[f"prob_{k}"] = probs[:, k]
    return table


def plot_curves(history: pd.DataFrame, title: str, path=None):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax_loss.plot(history["epoch"], history["train_loss"], label="train")
    ax_loss.plot(history["epoch"], history["val_loss"], label="validation")
    ax_loss.set(title="Loss", xlabel="epoch")
    ax_acc.plot(history["epoch"], history["train_acc"], label="train")
    ax_acc.plot(history["epoch"], history["val_acc"], label="validation")
    ax_acc.set(title="Accuracy", xlabel="epoch")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def run_experiment(model_name: str, build_model, split: pd.DataFrame, cache, band_stats,
                   input_type: str, seed: int, project_root, batch_size: int = 128,
                   max_epochs: int = 30, lr: float = 1e-3, weight_decay: float = 0.0,
                   patience: int = 5, device=None, verbose: bool = True) -> dict:
    # one full run (one input type, one seed). saves:
    #   results/predictions/<model>_<input>_seed<n>.csv  - test predictions
    #   results/training/<model>_<input>_seed<n>.csv     - loss/acc per epoch
    #   results/figures/curves_<model>_<input>_seed<n>.png
    #   models/<model>_<input>_seed<n>.pt                - weights (not in git)
    project_root = Path(project_root)
    run_name = f"{model_name}_{input_type}_seed{seed}"
    folders = {name: project_root / sub for name, sub in [
        ("predictions", "results/predictions"), ("training", "results/training"),
        ("figures", "results/figures"), ("models", "models")]}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    device = device or get_device()
    set_seed(seed)
    loaders = make_loaders(split, cache, band_stats, input_type, batch_size=batch_size, seed=seed)
    model = build_model(input_type)

    if verbose:
        print(f"{run_name} on {device}")
    start = time.time()
    history, best_epoch = fit(model, loaders["train"], loaders["val"], device,
                              max_epochs=max_epochs, lr=lr, weight_decay=weight_decay,
                              patience=patience, verbose=verbose)
    train_time = time.time() - start

    history.to_csv(folders["training"] / f"{run_name}.csv", index=False)
    plot_curves(history, run_name, folders["figures"] / f"curves_{run_name}.png")
    plt.close("all")
    torch.save(model.state_dict(), folders["models"] / f"{run_name}.pt")

    row = {"model": model_name, "input": input_type, "seed": seed,
           "best_epoch": best_epoch, "epochs_run": len(history),
           "train_time_s": round(train_time, 1)}
    for split_name in ["val", "test"]:
        rows = split[split["split"] == split_name]
        probs = predict_proba(model, loaders[split_name], device)
        y_true, y_pred = rows["label"].to_numpy(), probs.argmax(1)
        row[f"{split_name}_accuracy"] = accuracy_score(y_true, y_pred)
        row[f"{split_name}_macro_f1"] = f1_score(y_true, y_pred, average="macro",
                                                 labels=list(range(NUM_CLASSES)), zero_division=0)
        if split_name == "test":
            predictions_table(rows, probs).to_csv(folders["predictions"] / f"{run_name}.csv",
                                                  index=False)
    if verbose:
        print(f"  done in {train_time / 60:.1f} min, test acc {row['test_accuracy']:.3f}, "
              f"macro F1 {row['test_macro_f1']:.3f}")
    return row
