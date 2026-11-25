"""ResNet18 pretrained on ImageNet, for 3 (RGB) or 13 input bands. Used by notebooks 05 and 07.

Written as in the course's fine-tuning notebook (lecture_12_finetuning_live_WITH_RESULTS 1.pdf,
section 4, strategy C): ImageNet weights, a new 10-class head and full fine-tuning with a
10 times smaller learning rate for the pretrained backbone than for the new head.
"""

import torch
import torch.nn as nn
from torchvision import models

from .data import RGB_BAND_INDICES

IMAGE_SIZE = 224  # patches are resized from 64 x 64 to the ImageNet size before the network


def build_resnet(in_channels=3, num_classes=10, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    if in_channels == 13:
        # New first layer: the pretrained red, green and blue filters go to B04, B03 and B02,
        # the filters for the other 10 bands start at zero and are learned during fine-tuning.
        old = model.conv1
        model.conv1 = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.conv1.weight.zero_()
            model.conv1.weight[:, RGB_BAND_INDICES] = old.weight
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head
    return model


def resnet_param_groups(model, lr_backbone, lr_head):
    """Two optimizer groups as in the course: the backbone gets a smaller learning rate than the head."""
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    return [{"params": backbone, "lr": lr_backbone},
            {"params": model.fc.parameters(), "lr": lr_head}]
