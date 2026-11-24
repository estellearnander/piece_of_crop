"""The CNN of the course notebook (CNN_Class_Competition_2_Food_Classification_Explained.pdf,
section 2), with the number of input channels as an argument: 3 for RGB, 13 for all bands.

Used by notebook 04 (training) and notebook 07 (haze test). Only the first
convolution changes between RGB and 13 bands, so a difference in the results
cannot come from the architecture.
"""

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        # Input: (B, in_channels, 64, 64)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Input: (B, 32, 32, 32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Input: (B, 64, 16, 16)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.classifier = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)  # (B, 128, 1, 1)
        x = x.flatten(start_dim=1)  # (B, 128)
        return self.classifier(x)  # (B, 10) logits
