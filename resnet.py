import torch
import torch.nn as nn
from copy import deepcopy
import torch_directml as dml


class ResBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.bnorm1 = nn.BatchNorm2d(num_features)
        self.bnorm2 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_features, input_features, policy_size):
        super().__init__()
        self.start_block = nn.Sequential(
            nn.Conv2d(input_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList(
            [ResBlock(num_features) for _ in range(num_blocks)]
        )
        p_size, v_size = self.calculate_input_size(
            num_features, input_features)
        self.policy = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1),
            nn.Conv2d(num_features, num_features, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(p_size, policy_size)
        )
        self.value = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(v_size, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, 1),
            nn.Tanh()
        )
        self.__disable_policy = False

    def forward(self, x):
        x = self.start_block(x)
        for block in self.blocks:
            x = block(x)
        p = self.policy(x).squeeze()
        v = self.value(x).squeeze()
        if self.__disable_policy:
            return v
        return p, v

    def to_tensor(self, data):
        return torch.tensor(data).float().unsqueeze(0)

    def batch_to_tensor(self, data):
        return torch.tensor(data).float()

    def get_policy(self, p):
        return torch.softmax(p, 1).squeeze(0).detach().cpu().numpy()

    def get_value(self, v):
        return v.item()

    def calculate_input_size(self, num_features, input_features):
        x = torch.zeros((1, input_features, 8, 8)).float()
        start_block_copy = deepcopy(self.start_block).float()
        blocks_copy = deepcopy(self.blocks).float()
        p_seqence = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1),
            nn.Conv2d(num_features, num_features, 1),
            nn.Flatten()
        )
        v_seqence = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1),
            nn.Flatten()
        )

        x = start_block_copy(x)
        for block in blocks_copy:
            x = block(x)
        p = p_seqence(x)
        v = v_seqence(x)
        return p.shape[1], v.shape[1]

    def disable_policy(self):
        self.__disable_policy = True

    def enable_policy(self):
        self.__disable_policy = False
