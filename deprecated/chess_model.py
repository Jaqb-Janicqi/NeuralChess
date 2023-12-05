from copy import deepcopy
from se_resblock import SeResBlock
from resblock import ResBlock
import torch
import torch.nn as nn


class ChessModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.__num_blocks = kwargs["num_blocks"]
        self.__num_features = kwargs["num_features"]
        self.__num_input_features = kwargs["num_input_features"]
        self.__num_dense = kwargs["num_dense"]
        self.__dense_size = kwargs["dense_size"]
        self.__dtype = kwargs["dtype"]
        self.__squeeze_and_excitation = kwargs["squeeze_and_excitation"]

        # self.start_block = nn.Sequential(
        #     nn.Conv2d(self.__num_input_features, self.__num_features,
        #               3, 1, 1, dtype=self.__dtype),
        #     nn.BatchNorm2d(self.__num_features, dtype=self.__dtype),
        #     nn.ReLU(inplace=True)
        # )
        # if self.__squeeze_and_excitation:
        #     self.blocks = nn.Sequential(
        #         *[SeResBlock(self.__num_features, dtype=self.__dtype) for _ in range(self.__num_blocks)]
        #     )
        # else:
        #     self.blocks = nn.Sequential(
        #         *[ResBlock(self.__num_features, dtype=self.__dtype) for _ in range(self.__num_blocks)]
        #     )
        # self.to_dense = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.blocks_output_size(),
        #               self.__dense_size, dtype=self.__dtype),
        #     nn.ReLU(inplace=True)
        # )

        self.to_dense = nn.Linear(768, self.__dense_size, dtype=self.__dtype)
        dense_stack = []
        dense_size = self.__dense_size
        for _ in range(self.__num_dense):
            dense_out = int(dense_size * kwargs["dense_reduce_factor"])
            dense_stack.append(
                nn.Linear(dense_size, dense_out, dtype=self.__dtype))
            dense_stack.append(nn.ReLU(inplace=True))
            dense_size = dense_out
        self.dense = nn.Sequential(*dense_stack)
        # self.value = nn.Sequential(
        #     nn.Linear(dense_size, 1, dtype=self.__dtype),
        #     nn.Tanh()
        # )
        self.value = nn.Linear(dense_size, 1, dtype=self.__dtype)

    def blocks_output_size(self):
        x = torch.zeros((1, self.__num_input_features, 8, 8)).float()
        blocks_copy = deepcopy(self.blocks)
        flatten = nn.Flatten()

        with torch.no_grad():
            x = self.start_block(x)
            for block in blocks_copy:
                x = block(x)
            x = flatten(x)
        return x.shape[1]

    def forward(self, x):
        # x = self.start_block(x)
        # x = self.blocks(x)
        x = self.to_dense(x)
        x = self.dense(x)
        v = self.value(x)
        return v

    @property
    def num_blocks(self):
        return self.__num_blocks

    @property
    def num_features(self):
        return self.__num_features
