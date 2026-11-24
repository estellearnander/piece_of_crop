"""Small CNN trained from scratch, for 3 (RGB) or 13 input bands.

Used by notebook 04 (training) and notebook 07 (haze test). The two inputs use
exactly the same network: only the number of input channels changes, so a
difference between RGB and 13 bands cannot come from the architecture.
"""

import torch.nn as nn

INPUT_CHANNELS = {"rgb": 3, "13band": 13}


class SmallCNN(nn.Module):
    """Three Conv-BatchNorm-ReLU-MaxPool blocks (32, 64, 128 filters), then
    global average pooling and one linear layer with 10 outputs.

    A 64 x 64 patch becomes 32 x 32, 16 x 16 and 8 x 8 after the three poolings.
    The network returns logits; CrossEntropyLoss applies the softmax.
    """

    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        blocks = []
        channels = in_channels
        for out_channels in (32, 64, 128):
            blocks.extend([
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ])
            channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_cnn(input_type: str, num_classes: int = 10) -> SmallCNN:
    """Return a SmallCNN with 3 input channels for "rgb" or 13 for "13band"."""
    if input_type not in INPUT_CHANNELS:
        raise ValueError(f"input_type must be 'rgb' or '13band', not {input_type!r}")
    return SmallCNN(in_channels=INPUT_CHANNELS[input_type], num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
