import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.bnorm1 = nn.BatchNorm2d(num_features)
        self.bnorm2 = nn.BatchNorm2d(num_features)
        self.nonlinear = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

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
