# Tensor size is: [B = Batch_size, C = Channels, H = Height, W = Width]
# output_size = (dim_in - kernel_size + 2*padding)/stride + 1

import torch
import torch.nn as nn

class SimpleImageClassifier(nn.Module):
  def __init__(self, n_classes: int, dropout: float = 0.1):
    super().__init__()

    self.features = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels=16, kernel_size = 3, padding = 1), # from [B, 1, 28, 28] to [B, 16, 28, 28]
        nn.ReLU(),
        nn.Conv2d(in_channels = 16, out_channels=32, kernel_size = 3, padding = 1), # from [B, 16, 28, 28] to [B, 32, 28, 28]
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride = 2), # [B, 32, 14, 14]

        nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = 3, padding = 1), # from [B, 32, 28, 28] to [B, 64, 28, 28]
        nn.ReLU(),
        nn.Conv2d(in_channels = 64, out_channels=128, kernel_size = 3, padding = 1), # from [B, 64, 28, 28] to [B, 128, 28, 28]
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride = 2) # [B, 128, 7, 7]
    )

    self.flatten = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size = 2), # [B, 128, 2, 2]
        nn.Flatten(start_dim = 1), # [B, 128 * 2 * 2]
    )

    self.classifier = nn.Sequential(
        nn.Linear(in_features = 128 * 2 * 2, out_features = 128),
        nn.ReLU(),
        nn.Dropout(p = dropout),
        nn.Linear(in_features = 128, out_features = n_classes)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.flatten(x)
    logits = self.classifier(x)
    return logits

def build_model(num_classes: int, dropout: float = 0.1) -> nn.Module:
    return SimpleImageClassifier(n_classes=num_classes, dropout=dropout)