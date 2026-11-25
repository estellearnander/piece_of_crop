"""Training and evaluation for the CNN (notebook 04) and the ResNet (notebook 05).

Written as in the course: train_one_epoch and evaluate as in the CNN notebook
(CNN_Class_Competition_2_Food_Classification_Explained.pdf, section 4), and the
training recipe of the fine-tuning notebook (lecture_12_finetuning_live_WITH_RESULTS 1.pdf,
section 4): AdamW with weight decay, a cosine learning-rate schedule and a history
of loss and accuracy per epoch. In addition, the epoch with the lowest validation
loss is kept, so the validation set chooses the final model.
"""

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def seed_all(s=0):
    """Fix the random seeds, as in the course (lecture_12 live notebook, setup)."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, frozen_backbone=False):
    model.train()
    if frozen_backbone:  # keep the BatchNorm statistics of a pretrained backbone fixed
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, epochs, device, frozen_backbone=False):
    """Train for a fixed number of epochs with a cosine schedule and keep the best epoch.

    Returns (history, best_epoch). The model ends with the weights of the epoch
    with the lowest validation loss.
    """
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                                device, frozen_backbone)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}: train loss={train_loss:.3f} acc={train_acc:.3f} | "
              f"validation loss={val_loss:.3f} acc={val_acc:.3f}")
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, best_epoch


@torch.no_grad()
def predict(model, loader, device, zero_bands=None):
    """Return (true labels, class probabilities) for every patch of a loader, in loader order.

    zero_bands: optional list of input channels set to 0 before the forward pass. After
    normalisation 0 is the training mean, so the network gets no information from them.
    """
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        if zero_bands is not None:
            images = images.clone()
            images[:, zero_bands] = 0
        logits = model(images.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


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
