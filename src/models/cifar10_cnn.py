# cifar10_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cifar10CNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, dropout_p: float = 0):
        super().__init__()

        def conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, cout),
                nn.ReLU(inplace=False),
            )

        # Feature extractor
        self.block1 = nn.Sequential(
            conv_bn_relu(in_channels, 64),
            conv_bn_relu(64, 64),
            nn.MaxPool2d(2),         
            nn.Dropout(p=dropout_p),
        )

        self.block2 = nn.Sequential(
            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool2d(2),          
            nn.Dropout(p=dropout_p),
        )

        self.block3 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(2),          
            nn.Dropout(p=dropout_p),
        )

        # Head: global average pool + linear
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # (B, 256, 1, 1)
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        # Return flattened features before classifier
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)  # (B, 256)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = self.fc(x)
        return x
