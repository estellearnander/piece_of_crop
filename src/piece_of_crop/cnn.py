"""Small CNN trained from scratch (owner: Rishi).

3 conv blocks -> global avg pool -> linear. only in_channels changes between
rgb (3) and 13 bands so the comparison is fair
"""

import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    # 2 convs + batchnorm + relu, then maxpool halves the size
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    # 64x64 patch in, 10 class scores out

    def __init__(self, in_channels: int = 13, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            conv_block(in_channels, 32),   # 64 -> 32
            conv_block(32, 64),            # 32 -> 16
            conv_block(64, 128),           # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # 128 x 8 x 8 -> 128
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_cnn(input_type: str, num_classes: int = 10) -> SmallCNN:
    channels = {"rgb": 3, "13band": 13}
    if input_type not in channels:
        raise ValueError(f"input_type must be 'rgb' or '13band', not {input_type!r}")
    return SmallCNN(in_channels=channels[input_type], num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
