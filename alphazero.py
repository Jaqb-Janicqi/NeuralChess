import os
import sqlite3
import yaml
import tqdm

import numpy as np
import torch
import torch_directml as dml
import chess
from torch import nn
import pytorch_lightning as pl
from cache_read_priority import Cache
from db_dataloader import DataLoader
from mcts import MCTS
from resnet import ResNet
from actionspace import ActionSpace
import sys
from trainer import NetworkTrainer
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter


def convert_to_numpy(arr):
    return np.frombuffer(arr, dtype=np.float32).reshape(16, 8, 8)


class AlphaZero():
    def __init__(self, device=dml.device()) -> None:
        self.__load_args()
        self.__device = device
        self.__action_space = ActionSpace()
        self.__evaluation_cache = Cache(self.__training_args['cache_size'])
        self.__infinity = sys.maxsize ** 10

    def __load_args(self):
        with open("config.yaml", "r") as config_file:
            self.__args = yaml.safe_load(config_file)
        with open("training_config.yaml", "r") as training_config_file:
            self.__training_args = yaml.safe_load(training_config_file)

    def step(self, probs: np.ndarray):
        return self.__action_space[np.argmax(probs)]

    def stochastic_step(self, probs: np.ndarray):
        return self.__action_space[np.random.choice(
            self.__action_space.size, p=probs)]

    def train_value(self):
        conn = sqlite3.connect(self.__training_args['db_path'])
        db_size = conn.execute(
            "SELECT COUNT(*) FROM positions").fetchone()[0]
        conn.close()
        dataloader = DataLoader(
            db_path='C:/sqlite_chess_db/chess_positions.db',
            table_name='positions',
            num_batches=10000,
            batch_size=1024,
            min_index=1,
            max_index=db_size*0.5,
            random=True,
            replace=False,
            shuffle=True,
            slice_size=64,
            specials={'encoded': convert_to_numpy}
        )
        dataloader.start()
        val_dataloader = DataLoader(
            db_path='C:/sqlite_chess_db/chess_positions.db',
            table_name='positions',
            num_batches=1000,
            batch_size=1024,
            min_index=db_size*0.7,
            max_index=db_size,
            random=False,
            replace=False,
            shuffle=False,
            slice_size=64,
            specials={'encoded': convert_to_numpy}
        )
        val_dataloader.start()
        model = ResNet(
            self.__args['num_blocks'],
            self.__args['num_features'],
            self.__args['input_features'],
            self.__action_space.size
        )
        model.to(self.__device)
        trainer = NetworkTrainer(model, True, self.__device)
        trainer.fit(dataloader, val_dataloader, 30, True)

    def find_lr(self):
        conn = sqlite3.connect(self.__training_args['db_path'])
        db_size = conn.execute(
            "SELECT COUNT(*) FROM positions").fetchone()[0]
        conn.close()
        tmp = DataLoader(
            db_path='C:/sqlite_chess_db/chess_positions.db',
            table_name='positions',
            num_batches=0,
            batch_size=1024,
            min_index=1,
            max_index=db_size,
            random=True,
            replace=True,
            shuffle=True,
            slice_size=64,
            specials={'encoded': convert_to_numpy},
            data_cols=['encoded'],
            label_cols=['prob']
        )
        tmp.start()
        model = ResNet(
            self.__args['num_blocks'],
            self.__args['num_features'],
            self.__args['input_features'],
            self.__action_space.size
        )
        model.disable_policy()
        model.to(self.__device)
        trainer = NetworkTrainer(model, True, self.__device)
        trainer.lr_finder(tmp, 1e-5, 0.1, 0.0015, 100)

if __name__ == "__main__":
    az = AlphaZero()
    # az.find_lr()
    az.train_value()
