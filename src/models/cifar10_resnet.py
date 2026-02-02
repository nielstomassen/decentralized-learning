import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def make_gn(num_channels: int, num_groups: int = 8):
    # groups must divide channels; 8 works for 16/32/64, etc.
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.norm1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, 1)
        self.norm2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class CifarResNet(nn.Module):
    def __init__(self, num_classes=10, n=5, in_channels=3, norm="bn", gn_groups=8):
        super().__init__()
        self.in_planes = 16

        if norm == "gn":
            norm_layer = lambda c: nn.GroupNorm(num_groups=gn_groups, num_channels=c)
        else:
            norm_layer = nn.BatchNorm2d

        self.norm_layer = norm_layer

        self.conv1 = conv3x3(in_channels, 16, stride=1)
        self.norm1 = norm_layer(16)

        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s, norm_layer=self.norm_layer))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def feature(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.feature(x)
        return self.fc(x)

def resnet32_cifar_gn(num_classes=10, gn_groups=8):
    return CifarResNet(num_classes=num_classes, n=5, norm="gn", gn_groups=gn_groups)

def resnet32_cifar_bn(num_classes=10):
    return CifarResNet(num_classes=num_classes, n=5, norm="bn")
