# cifar10_cnn.py

import torch.nn as nn
import torch.nn.functional as F

class Cifar10CNN(nn.Module):
    """
    Small CNN for 32x32 RGB images (e.g. CIFAR-10).

    Architecture (fairly standard):
    - Conv(3 -> 32), ReLU, MaxPool
    - Conv(32 -> 64), ReLU, MaxPool
    - Conv(64 -> 128), ReLU, MaxPool
    - FC: 128 * 4 * 4 -> 256, ReLU
    - FC: 256 -> 10
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            padding=1
        )  # output: 32 x 32 x 32
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )  # output: 64 x 16 x 16 (after pool)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )  # output: 128 x 8 x 8 (after pool)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves H and W

        # After three pooling layers: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch, in_channels, 32, 32)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)      # (batch, 32, 16, 16)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)      # (batch, 64, 8, 8)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)      # (batch, 128, 4, 4)

        x = x.view(x.size(0), -1)  # flatten: (batch, 128*4*4)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)            # logits (batch, num_classes)
        return x
