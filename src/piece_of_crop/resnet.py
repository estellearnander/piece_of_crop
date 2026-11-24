"""ResNet18 pretrained on ImageNet, adapted to 3 (RGB) or 13 input bands.

Used by notebook 05 (fine-tuning) and notebook 07 (haze test).
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet(input_type: str, pretrained: bool = True, num_classes: int = 10) -> nn.Module:
    """Return a ResNet18 for "rgb" or "13band" input with a new 10-class head.

    13 bands: conv1 gets 13 input channels. The three pretrained ImageNet filters
    (red, green, blue) are copied to the positions of B04, B03 and B02, and the
    weights for the other 10 bands start at zero and are learned during fine-tuning.

    Selective fine-tuning: everything is frozen except the last residual stage
    (layer4), the new head (fc) and, for 13 bands, the new conv1.
    Use pretrained=False only for shape tests and for loading saved weights.
    """
    if input_type not in ("rgb", "13band"):
        raise ValueError("input_type must be 'rgb' or '13band'")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = resnet18(weights=weights)
    if input_type == "13band":
        old = net.conv1
        new = nn.Conv2d(13, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, 3] = old.weight[:, 0]  # B04 = red
            new.weight[:, 2] = old.weight[:, 1]  # B03 = green
            new.weight[:, 1] = old.weight[:, 2]  # B02 = blue
        net.conv1 = new
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in net.layer4.parameters():
        parameter.requires_grad_(True)
    for parameter in net.fc.parameters():
        parameter.requires_grad_(True)
    if input_type == "13band":
        for parameter in net.conv1.parameters():
            parameter.requires_grad_(True)
    return net


def resnet_param_groups(net: nn.Module, input_type: str, lr_backbone: float, lr_head: float) -> list:
    """Optimizer groups: a smaller learning rate for layer4 (and conv1) than for the new head."""
    groups = [
        {"params": net.layer4.parameters(), "lr": lr_backbone},
        {"params": net.fc.parameters(), "lr": lr_head},
    ]
    if input_type == "13band":
        groups.append({"params": net.conv1.parameters(), "lr": lr_backbone})
    return groups

