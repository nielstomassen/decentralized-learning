# mnist_cnn.py

import torch.nn as nn
import torch.nn.functional as F

class MnistCNN(nn.Module):
    """
    Small CNN for 28x28 grayscale images (e.g. MNIST).

    Architecture:
    - Conv(1 -> 32), ReLU, MaxPool
    - Conv(32 -> 64), ReLU, MaxPool
    - FC: 64 * 7 * 7 -> 128, ReLU
    - FC: 128 -> 10
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)   # output: 32 x 28 x 28
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)   # output: 64 x 14 x 14 (after pool)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves H and W

        # After two pooling layers: 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, in_channels, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)     # (batch, 32, 14, 14)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)     # (batch, 64, 7, 7)

        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)            # logits (batch, num_classes)
        return x
