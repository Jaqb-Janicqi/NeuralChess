from copy import deepcopy
from resnet.se_resblock import SeResBlock
from resnet.resblock import ResBlock
import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if kwargs.get("dtype") is None:
            self.__dtype = torch.float32
        else:
            self.__dtype = kwargs.get("dtype")
        if kwargs.get("device") is None:
            self.__device = torch.device("cpu")
        else:
            self.__device = kwargs.get("device")
        self.__num_blocks = kwargs.get("num_blocks")
        self.__num_features = kwargs.get("num_features")
        self.__num_input_features = kwargs.get("num_input_features")
        self.__policy_size = kwargs.get("policy_size")
        self.__squeeze_and_excitation = kwargs.get("squeeze_and_excitation")
        self.__weight_init_mode = kwargs.get("weight_init_mode")
        self.__disable_policy = False
        self.__init_model()

    def __init_model(self):
        self.start_block = nn.Sequential(
            nn.Conv2d(self.__num_input_features,
                      self.__num_features, 1, dtype=self.__dtype),
            nn.BatchNorm2d(self.__num_features, dtype=self.__dtype),
            nn.ReLU(inplace=True)
        )
        if self.__squeeze_and_excitation:
            self.blocks = nn.ModuleList(
                [SeResBlock(self.__num_features, dtype=self.__dtype)
                 for _ in range(self.__num_blocks)]
            )
        else:
            self.blocks = nn.ModuleList(
                [ResBlock(self.__num_features, dtype=self.__dtype)
                 for _ in range(self.__num_blocks)]
            )
        p_size, v_size = self.calculate_input_size()
        self.policy = nn.Sequential(
            nn.Conv2d(self.__num_features, self.__num_features,
                      1, dtype=self.__dtype),
            nn.Conv2d(self.__num_features, 80, 1, dtype=self.__dtype),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(p_size, self.__policy_size, dtype=self.__dtype),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Conv2d(self.__num_features, int(
                self.__num_features/2), 1, dtype=self.__dtype),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(v_size, 3, dtype=self.__dtype),
            nn.Linear(3, 1, dtype=self.__dtype),
            nn.Tanh()
        )
        if self.__weight_init_mode is not None:
            self.init_weights(self.__weight_init_mode)

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

    def calculate_input_size(self):
        x = torch.zeros((1, self.__num_input_features, 8, 8)).float()
        start_block_copy = deepcopy(self.start_block).float()
        blocks_copy = deepcopy(self.blocks).float()
        p_seqence = nn.Sequential(
            nn.Conv2d(self.__num_features, self.__num_features, 1),
            nn.Conv2d(self.__num_features, 80, 1),
            nn.Flatten(),
        )
        v_seqence = nn.Sequential(
            nn.Conv2d(self.__num_features, int(self.__num_features/2), 1),
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

    @property
    def num_blocks(self):
        return self.__num_blocks

    @property
    def num_features(self):
        return self.__num_features

    @property
    def input_features(self):
        return self.__input_features

    @property
    def policy_size(self):
        return self.__policy_size

    def to(self, device):
        self.__device = device
        return super().to(device)
