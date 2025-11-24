# cifar10_logreg.py

import torch
import torch.nn as nn

class Cifar10LogReg(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, img_size: int = 32):
        """
        Simple logistic regression for CIFAR-10.

        Flattens the 3x32x32 image to a vector and applies a single linear layer.
        """
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.num_classes = num_classes

        self.fc = nn.Linear(in_channels * img_size * img_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, in_channels, H, W)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        return out
