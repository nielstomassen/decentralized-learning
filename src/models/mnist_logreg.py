# mnist_logreg.py
import torch
import torch.nn as nn

class MnistLogReg(nn.Module):
    """
    Logistic regression model for 28x28 grayscale images (e.g. MNIST).

    This is just a single linear layer:
    - Flatten 28x28 -> 784
    - Linear: 784 -> num_classes

    No hidden layers, no non-linearities.
    """

    def __init__(self, in_channels: int = 1, img_size: int = 28,
                 num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.num_classes = num_classes
        # Total number of input features per image
        in_features = in_channels * img_size * img_size  # 1 * 28 * 28 = 784

        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        x: (batch_size, in_channels, H, W) = (N, 1, 28, 28)
        returns: logits of shape (batch_size, num_classes)
        """
        # Flatten all dimensions except batch
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)          
        return x
