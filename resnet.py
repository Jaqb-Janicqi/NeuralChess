from copy import deepcopy
from typing import Union

from torch import device
from se_resblock import SeResBlock
from resblock import ResBlock
import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_features, input_features, policy_size, se=False, dtype=torch.float32, device=torch.device("cpu")):
        super().__init__()
        self.__dtype = dtype
        self.__device = device
        self.start_block = nn.Sequential(
            nn.Conv2d(input_features, num_features, 3, 1, 1, dtype=dtype),
            nn.BatchNorm2d(num_features, dtype=dtype),
            nn.ReLU(inplace=True)
        )
        if se:  # Squeeze and Excitation
            self.blocks = nn.ModuleList(
                [SeResBlock(num_features, dtype=dtype)
                 for _ in range(num_blocks)]
            )
        else:
            self.blocks = nn.ModuleList(
                [ResBlock(num_features, dtype=dtype)
                 for _ in range(num_blocks)]
            )
        p_size, v_size = self.calculate_input_size(
            num_features, input_features)
        self.policy = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1, dtype=dtype),
            nn.Conv2d(num_features, 80, 1, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(p_size, policy_size, dtype=dtype),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Conv2d(num_features, 80, 1, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(v_size, 256, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1, dtype=dtype),
            nn.Tanh()
        )
        self.__disable_policy = False
        self.__squeeze_and_excitation = se
        nn.init.xavier_uniform_(self.start_block[0].weight)
        nn.init.xavier_uniform_(self.policy[0].weight)
        nn.init.xavier_uniform_(self.policy[1].weight)
        nn.init.xavier_uniform_(self.value[0].weight)
        nn.init.xavier_uniform_(self.value[3].weight)
        nn.init.xavier_uniform_(self.value[5].weight)

    def forward(self, x):
        x = self.start_block(x)
        for block in self.blocks:
            x = block(x)
        v = self.value(x)
        if self.__disable_policy:
            return v
        p = self.policy(x)
        return p, v

    def to_tensor(self, data):
        return torch.tensor(data).to(self.dtype).unsqueeze(0).to(self.__device)

    def batch_to_tensor(self, data):
        return torch.tensor(data).to(self.dtype).to(self.__device)

    def get_policy(self, p):
        return p.detach().cpu().numpy()

    def get_value(self, v):
        return v.item()

    def calculate_input_size(self, num_features, input_features):
        x = torch.zeros((1, input_features, 8, 8)).float()
        start_block_copy = deepcopy(self.start_block).float()
        blocks_copy = deepcopy(self.blocks).float()
        p_seqence = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1),
            nn.Conv2d(num_features, 80, 1),
            nn.Flatten(),
        )
        v_seqence = nn.Sequential(
            nn.Conv2d(num_features, 80, 1),
            nn.Flatten(),
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

    @property
    def policy_disabled(self):
        return self.__disable_policy

    @property
    def squeeze_and_excitation(self):
        return self.__squeeze_and_excitation

    @property
    def dtype(self):
        return self.__dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype == torch.float16:
            self.half()
        elif dtype == torch.float32:
            self.float()
        self.__dtype = dtype

    def to(self, device):
        self.__device = device
        return super().to(device)


if __name__ == "__main__":
    import torch.nn.functional as F
    net = ResNet(20, 256, 8, 1968)
    policy, val = net(torch.zeros((1, 8, 8, 8)))
    zeros = torch.zeros(1)
    val = val.flatten()
    loss = F.mse_loss(val, zeros)
    policy_loss = F.cross_entropy(policy, torch.zeros((1, 1968)))
    loss = loss + policy_loss
    loss.backward()
    policy = net.get_policy(policy)
    val = net.get_value(val)
    print(val)
    print(policy.shape)
