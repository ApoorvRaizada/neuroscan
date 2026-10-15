"""Baseline CNN and ResNet18 builder."""
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """Small 3-block CNN trained from scratch. The bar ResNet18 has to beat."""

    def __init__(self, num_classes: int = 4):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(num_classes: int = 4, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m