import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet5 for 28x28 grayscale (MNIST). ~60K params.
    Matches paper: 2 conv layers + 3 FC layers.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 28x28 -> 28x28
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                          # 28x28 -> 14x14
            nn.Conv2d(6, 16, kernel_size=5),             # 14x14 -> 10x10
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                          # 10x10 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LogisticRegression(nn.Module):
    """Logistic regression for Spambase (57 features, binary)."""

    def __init__(self, in_features: int = 57, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
