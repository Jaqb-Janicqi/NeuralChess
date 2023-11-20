import torch.nn as nn
from torchvision.ops import SqueezeExcitation


class SeResBlock(nn.Module):
    def __init__(self, num_features, reduction=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.bnorm1 = nn.BatchNorm2d(num_features)
        self.bnorm2 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcitation(num_features, reduction)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.se(x)
        x += residual
        x = self.relu(x)
        return x
