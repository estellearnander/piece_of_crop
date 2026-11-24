"""Training, evaluation and saving predictions, shared by the CNN (notebook 04)
and the ResNet (notebook 05).

Both models use the same loop: cross-entropy loss, one optimizer step per batch,
evaluation on the validation set after every epoch, and early stopping on the
validation loss. The best epoch (lowest validation loss) is restored at the end,
and only then is the test set evaluated.
"""

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score


LABEL_IDS = np.arange(10)


def set_seed(seed: int) -> None:
    """Fix the random seeds of Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """CUDA if available, otherwise the Mac GPU (MPS), otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resize_images(images: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a batch [B, C, 64, 64] to [B, C, size, size] with bilinear interpolation (ResNet input)."""
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def _prepare(images, labels, device, input_size):
    images = images.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    if input_size is not None:
        images = resize_images(images, input_size)
    return images, labels


def train_epoch(model, loader, optimizer, loss_fn, device, input_size=None, freeze_batchnorm=False):
    """Train for one epoch. Returns (mean training loss, training accuracy).

    input_size: resize the patches to this size first (ResNet), or None (CNN).
    freeze_batchnorm: keep the BatchNorm running statistics of a pretrained
    backbone fixed by putting those layers in evaluation mode (ResNet).
    """
    model.train()
    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    total_loss, total_n, correct = 0.0, 0, 0
    for images, labels in loader:
        images, labels = _prepare(images, labels, device, input_size)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        n = len(labels)
        total_loss += loss.item() * n
        total_n += n
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / total_n, correct / total_n


@torch.inference_mode()
def evaluate_model(model, loader, loss_fn, device, input_size=None):
    """Loss, accuracy, macro-F1, labels, predictions and class probabilities on one loader."""
    model.eval()
    total_loss, total_n = 0.0, 0
    truths, probabilities = [], []
    for images, labels in loader:
        images, labels = _prepare(images, labels, device, input_size)
        logits = model(images)
        n = len(labels)
        total_loss += loss_fn(logits, labels).item() * n
        total_n += n
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        truths.append(labels.cpu().numpy())
    y_true = np.concatenate(truths)
    proba = np.concatenate(probabilities)
    y_pred = proba.argmax(axis=1)
    return {
        "loss": total_loss / total_n,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
        "proba": proba,
    }


def fit(model, loaders, optimizer, device, max_epochs, patience, run_info,
        input_size=None, freeze_batchnorm=False):
    """Train with early stopping on the validation loss and restore the best epoch.

    run_info: dict with model, input and seed, copied into every history row.
    Returns (history as a list of dicts, the loss function).
    """
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = train_epoch(model, loaders["train"], optimizer, loss_fn,
                                                 device, input_size, freeze_batchnorm)
        val = evaluate_model(model, loaders["val"], loss_fn, device, input_size)
        history.append({**run_info, "epoch": epoch,
                        "train_loss": train_loss, "train_accuracy": train_accuracy,
                        "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                        "val_macro_f1": val["macro_f1"]})
        print(f"{run_info['input']} seed={run_info['seed']} epoch={epoch:02d} "
              f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
              f"val_acc={val['accuracy']:.3f} val_F1={val['macro_f1']:.3f}", flush=True)
        if val["loss"] < best_loss - 1e-4:
            best_loss = val["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping on validation loss.")
                break
    if best_state is None:
        raise RuntimeError("No validation checkpoint could be selected.")
    model.load_state_dict(best_state)
    return history, loss_fn


def predictions_table(files, y_true, proba) -> pd.DataFrame:
    """The team format for test predictions: file, y_true, y_pred, confidence, p_0 ... p_9.

    y_pred is the class with the highest probability and confidence is that probability.
    Used by notebooks 03, 04 and 05, so every model saves exactly the same columns.
    """
    proba = np.asarray(proba, dtype=float)
    table = pd.DataFrame({
        "file": np.asarray(files),
        "y_true": np.asarray(y_true, dtype=int),
        "y_pred": proba.argmax(axis=1),
        "confidence": proba.max(axis=1),
    })
    for class_id in range(proba.shape[1]):
        table[f"p_{class_id}"] = proba[:, class_id]
    return table


def check_test_order(split: pd.DataFrame, y_true) -> None:
    """Stop if the test loader's labels are not in the row order of split.csv.

    Predictions are saved next to the file names from split.csv, so a different
    order would attach predictions to the wrong patches.
    """
    expected = split.loc[split["split"].eq("test"), "label"].to_numpy(dtype=int)
    if not np.array_equal(np.asarray(y_true, dtype=int), expected):
        raise ValueError("Test loader order differs from split.csv.")
