from copy import deepcopy
from se_resblock import SeResBlock
from resblock import ResBlock
import torch
import torch.nn as nn
torch.manual_seed(0)


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_features, input_features, policy_size, se=False):
        super().__init__()
        self.start_block = nn.Sequential(
            nn.Conv2d(input_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        if se: # Squeeze and Excitation
            self.blocks = nn.ModuleList(
                [SeResBlock(num_features) for _ in range(num_blocks)]
            )
        else:
            self.blocks = nn.ModuleList(
                [ResBlock(num_features) for _ in range(num_blocks)]
            )
        p_size, v_size = self.calculate_input_size(
            num_features, input_features)
        self.policy = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1),
            nn.Conv2d(num_features, 128, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(p_size, policy_size),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Conv2d(num_features, 64, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(v_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        self.__disable_policy = False

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
        return torch.tensor(data).float().unsqueeze(0)

    def batch_to_tensor(self, data):
        return torch.tensor(data).float()

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
            nn.Conv2d(num_features, 128, 1),
            nn.Flatten(),
        )
        v_seqence = nn.Sequential(
            nn.Conv2d(num_features, 64, 1),
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


if __name__ == "__main__":
    import torch.nn.functional as F
    net = ResNet(20, 256, 8, 1968)
    policy, val = net(torch.zeros((4, 8, 8, 8)))
    zeros = torch.zeros(1)
    val = val.flatten()
    loss = F.mse_loss(val, zeros)
    policy_loss = F.cross_entropy(policy, torch.zeros((4, 1968)))
    loss = loss + policy_loss
    loss.backward()
    policy = net.get_policy(policy)
    val = net.get_value(val)
    print(policy.shape)