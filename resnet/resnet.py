from copy import deepcopy
from resnet.resblock import ResBlock
import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__num_blocks = kwargs.get("num_blocks")
        self.__num_features = kwargs.get("num_features")
        self.__num_input_features = kwargs.get("num_input_features")
        self.__weight_init_mode = kwargs.get("weight_init_mode")
        self.__init_model()

    def __init_model(self):
        self.start_block = nn.Sequential(
            nn.Conv2d(self.__num_input_features, self.__num_features, 1),
            nn.BatchNorm2d(self.__num_features),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList(
            [ResBlock(self.__num_features)
                for _ in range(self.__num_blocks)]
        )
        v_size = self.calculate_value_input_size()
        self.value = nn.Sequential(
            nn.Conv2d(self.__num_features, int(
                self.__num_features/2), 1),
            nn.BatchNorm2d(int(self.__num_features/2)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(v_size, 3),
            nn.Linear(3, 1),
            nn.Tanh()
        )
        if self.__weight_init_mode is not None:
            self.init_weights(self.__weight_init_mode)

    def forward(self, x):
        x = self.start_block(x)
        for block in self.blocks:
            x = block(x)
        return self.value(x)

    def to_tensor(self, data):
        return torch.tensor(data).unsqueeze(0).to(self.start_block[0].weight.device)

    def batch_to_tensor(self, data):
        return torch.tensor(data).to(self.start_block[0].weight.device)

    def get_value(self, v):
        return v.item()

    def init_weights(self, mode):
        if mode == "xavier":
            self.apply(self.__xavier_init)
        elif mode == "kaiming":
            self.apply(self.__kaiming_init)
        elif mode == "orthogonal":
            self.apply(self.__orthogonal_init)
        else:
            raise ValueError("Invalid weight initialization mode.")

    def __xavier_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def __kaiming_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def __orthogonal_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def calculate_value_input_size(self):
        x = torch.zeros((1, self.__num_input_features, 8, 8)).float()
        start_block_copy = deepcopy(self.start_block).float()
        blocks_copy = deepcopy(self.blocks).float()
        v_seqence = nn.Sequential(
            nn.Conv2d(self.__num_features, int(self.__num_features/2), 1),
            nn.Flatten(),
        )

        x = start_block_copy(x)
        for block in blocks_copy:
            x = block(x)
        v = v_seqence(x)
        return v.shape[1]

    @property
    def num_blocks(self):
        return self.__num_blocks

    @property
    def num_features(self):
        return self.__num_features

    @property
    def input_features(self):
        return self.__num_input_features