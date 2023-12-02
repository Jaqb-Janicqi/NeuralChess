import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, num_features, dtype=torch.float32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1, dtype=dtype)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1, dtype=dtype)
        self.bnorm1 = nn.BatchNorm2d(num_features, dtype=dtype)
        self.bnorm2 = nn.BatchNorm2d(num_features, dtype=dtype)
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x += residual
        x = self.nonlinear(x)
        return x
