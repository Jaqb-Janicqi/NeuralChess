import copy
import os
import pickle
import sqlite3
import time
import matplotlib.pyplot as plt
import yaml
import tqdm
import random
import cProfile

import numpy as np
import torch
import torch_directml
import chess
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
from adabound import AdaBound

from cache_read_priority import Cache
from mcts import MCTS
from resnet import ResNet
from actionspace import ActionSpace
from state import State
import sys


class AlphaZero():
    def __init__(self, device="cpu") -> None:
        self.__load_args()
        self.__action_space = ActionSpace()
        self.__board = chess.Board()
        self.__model = ResNet(
            self.__args['num_blocks'],
            self.__args['num_features'],
            self.__args['input_features'],
            self.__action_space.size,
            device
        )
        self.__optimizer = torch.optim.ASGD(
            self.__model.parameters(),
            lr=self.__training_args['lr']
        )
        # self.__optimizer = AdaBound(
        #     self.__model.parameters(),
        #     amsbound=True,
        # )
        self.__model_num = 1
        self.__value_loss_fn = nn.MSELoss()
        self.__policy_loss_fn = nn.CrossEntropyLoss()
        self.__evaluation_cache = Cache(self.__training_args['cache_size'])
        self.__infinity = sys.maxsize ** 10

    def __load_args(self):
        with open("config.yaml", "r") as config_file:
            self.__args = yaml.safe_load(config_file)
        with open("training_config.yaml", "r") as training_config_file:
            self.__training_args = yaml.safe_load(training_config_file)

    def pit(self, num_games, model1, model2) -> float:
        p1 = MCTS(self.__args, self.__action_space, Cache(), model1)
        p2 = MCTS(self.__args, self.__action_space, Cache(), model2)
        players = {
            1: {
                "model": p1,
                "wins": 0,
                "draws": 0,
                "losses": 0
            },
            -1: {
                "model": p2,
                "wins": 0,
                "draws": 0,
                "losses": 0
            }
        }
        for _ in range(num_games):
            while not self.__board.is_game_over():
                pass
        # return wr

    def step(self, probs: np.ndarray):
        return self.__action_space[np.argmax(probs)]

    def stochastic_step(self, probs: np.ndarray):
        return self.__action_space[np.random.choice(
            self.__action_space.size, p=probs)]

    def play_games(self):
        def search(mcts: MCTS):
            for i in range(self.__training_args['num_searches']):
                mcts.search_step(self.__infinity)

        self.__model.eval()
        dataset = []
        board = chess.Board()
        state = State(board)
        mcts = MCTS(self.__args, self.__action_space,
                    self.__evaluation_cache, self.__model)
        mcts.set_root(state.copy())
        mcts.initialize_root()
        for _ in range(self.__training_args['num_games']):
            while not mcts.root.state.is_terminal:
                # tic = time.time()
                with torch.no_grad():
                    search(mcts)
                # toc = time.time()
                # print(f"Search time: {toc - tic}")
                # print(f"Nodes per second: {self.__training_args['num_searches'] / (toc - tic)}")
                probs = mcts.get_dist()
                dataset.append((mcts.root.state.encoded, probs, np.array([mcts.evaluation])))
                action = chess.Move.from_uci(self.stochastic_step(probs))
                mcts.select_child_as_new_root(action)
                if len(dataset) > 2:
                    break
            if len(dataset) > 2:
                break
            mcts.reset_node_cache()
        return dataset

    def execute_epoch(self, dataloader, model, epoch_num=0):
        model.train()
        loss_history = []
        for state, policy, value in dataloader:
            state = self.__model.to_self(state)
            policy = self.__model.to_self(policy)
            value = self.__model.to_self(value)
            policy_pred, value_pred = model(state)
            policy_loss = self.__policy_loss_fn(policy_pred, policy)
            value_loss = self.__value_loss_fn(value_pred, value)
            loss = policy_loss + value_loss
            loss_history.append(loss.item())
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
        self.save_loss_plot(loss_history, epoch_num, "train")
        return model

    def train(self, window_size=100000, load_data=True):
        if load_data:
            # check if file exists
            if not os.path.isfile("train_data.pkl"):
                print("No train data file found.")
                train_data = []
            else:
                train_data = pickle.load(open("train_data.pkl", "rb"))

        for episode in trange(self.__training_args['num_episodes'], desc="Episodes"):
            self.__model.train()
            # play games and collect data

            # prepare dataloader
            train_data.extend(self.play_games())
            if len(train_data) > window_size:
                train_data = train_data[-window_size:]
            dataloader = DataLoader(
                train_data, batch_size=self.__training_args['batch_size'], shuffle=True)
            new_model = copy.deepcopy(self.__model)
            # backup optimiser to restore it if the new model is not better
            old_optimizer = self.__optimizer.state_dict()
            optimiser_restore = True
            for epoch in trange(self.__training_args['num_epochs'], desc="Epochs"):
                # train the model
                new_model = self.execute_epoch(dataloader, new_model, episode)
                wr = self.pit(
                    self.__training_args['num_pit_games'], self.__model, new_model)
                if wr > self.__training_args['win_rate_threshold']:
                    self.__model = new_model
                    self.__model_num += 1
                    self.__evaluation_cache.clear()
                    optimiser_restore = False
            if optimiser_restore:
                self.__optimizer.load_state_dict(old_optimizer)

        self.saturate()

    def save_loss_plot(self, loss_history, epoch_num, name):
        plt.plot(loss_history)
        plt.savefig(f"img/loss_plot_{name}_{epoch_num}.png")
        plt.close()

    def rand_offset(self, db_size, batch_size):
        return np.random.randint(0, db_size / batch_size) * batch_size

    def get_pre_train_batch(self, conn, batch_size, offset):
        try:
            batch = conn.execute(
                f"SELECT * FROM positions LIMIT {batch_size} OFFSET {offset}")
            batch = batch.fetchall()
            return batch
        except Exception as e:
            print(e)
            return None

    def create_pre_training_dataset(self, data, batch_size):
        dataset = []
        for id, fen, cp, prob in data:
            state = State(chess.Board(fen))
            state = self.__model.get_tensor_state(state.encoded)
            value = np.array([prob])
            dataset.append((state, value))
        dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataset

    def get_random_batch(self, request_size, batch_size, idx):
        conn = sqlite3.connect(self.__training_args['db_path'])
        # get a random batch
        batch = self.get_pre_train_batch(conn, request_size, idx)
        if batch is None:
            return None
        dataset = self.create_pre_training_dataset(batch, batch_size)
        conn.close()
        return dataset

    def get_train_test_idx(self, num_batches, batch_size):
        training_split = self.__training_args['training_split']
        if training_split is None:
            training_split = 0.9
        training_split = int(num_batches * training_split)
        idx = np.arange(num_batches)
        np.random.shuffle(idx)
        train_idx = idx[:training_split] * batch_size
        test_idx = idx[training_split:] * batch_size
        return train_idx, test_idx

    def pre_train_value(self):
        # connect to database and get its size
        conn = sqlite3.connect(self.__training_args['db_path'])
        if self.__training_args['db_size'] is None:
            db_size = conn.execute(
                "SELECT COUNT(*) FROM positions").fetchone()[0]
        else:
            db_size = self.__training_args['db_size']
        batch_size = self.__training_args['batch_size']
        db_request_size = self.__training_args['db_request_size']
        num_requests = db_size // db_request_size
        num_batches = db_request_size // batch_size
        max_batches = self.__training_args['max_batches_per_epoch']
        max_batches = min(max_batches, num_requests)
        train_idx_list, test_idx_list = self.get_train_test_idx(
            num_requests, db_request_size)
        best_model_num = None
        best_model_loss = np.inf

        # pre-train the value head
        num_epochs = self.__training_args['num_epochs_pre']
        train_loss_history = [[] for _ in range(num_epochs)]
        test_loss_history = [[] for _ in range(num_epochs)]
        for epoch in range(num_epochs):
            # shuffle indexes to rotate dataset if max_batches is set
            np.random.shuffle(train_idx_list)
            train_loss_history[epoch] = []
            test_loss_history[epoch] = []
            avg_model_test_loss = 0
            avg_model_train_loss = 0
            train_batches_num = len(train_idx_list)
            if self.__training_args['random_lr'] is not None:
                if epoch > self.__training_args['random_lr_start'] and self.__training_args['random_lr']:
                    lr_lower = self.__training_args['target_lr']
                    lr_upper = self.__training_args['lr']
                    self.__optimizer.param_groups[0]['lr'] = np.random.uniform(
                        lr_lower, lr_upper)

            pbar = tqdm.tqdm(total=max_batches,
                             desc="Epoch " + str(epoch) + ". Training.")
            for batch_num, train_idx in enumerate(train_idx_list):
                if batch_num >= max_batches:
                    train_batches_num = batch_num
                    break

                dataset = self.get_random_batch(
                    db_request_size, batch_size, train_idx)

                # train the model
                self.__model.train()
                train_loss = 0
                for state, value in dataset:
                    state = self.__model.to_self(state)
                    value = self.__model.to_self(value)
                    policy_pred, value_pred = self.__model(state)
                    value_loss = self.__value_loss_fn(value_pred, value)
                    self.__optimizer.zero_grad()
                    value_loss.backward()
                    self.__optimizer.step()
                    train_loss += value_loss.item()
                train_loss /= num_batches
                avg_model_train_loss += train_loss
                train_loss_history[epoch].append(
                    avg_model_train_loss / (batch_num + 1))
                pbar.set_description("Epoch " + str(epoch) + ". Training." +
                                     f" Avg Train Loss: {avg_model_train_loss / (batch_num + 1)}")
                pbar.update(1)

            test_loss = 0
            pbar.reset()
            pbar.set_description("Epoch " + str(epoch) + ". Testing.")
            pbar.total = len(test_idx_list)
            self.__model.eval()
            with torch.no_grad():
                for batch_num, test_idx in enumerate(test_idx_list):
                    if batch_num >= max_batches:
                        train_batches_num = batch_num
                        break
                    # get batch from the list
                    batch = self.get_pre_train_batch(
                        conn, batch_size, test_idx)
                    if batch is None:
                        continue
                    test_dataset = self.create_pre_training_dataset(
                        batch, batch_size)
                    for state, value in test_dataset:
                        state = self.__model.to_self(state)
                        value = self.__model.to_self(value)
                        policy_pred, value_pred = self.__model(state)
                        value_loss = self.__value_loss_fn(value_pred, value)
                        test_loss += value_loss.item()
                    test_loss /= num_batches
                    avg_model_test_loss += test_loss
                    test_loss_history[epoch].append(
                        avg_model_test_loss / (batch_num + 1))
                    pbar.update(1)
            avg_model_train_loss /= train_batches_num
            avg_model_test_loss /= train_batches_num
            tqdm.tqdm.write(
                f"Epoch {epoch} - Train Loss: {avg_model_train_loss} - Test Loss: {avg_model_test_loss}")

            # save the best model
            # if avg_model_test_loss < best_model_loss:
            #     best_model_loss = avg_model_test_loss
            model_name = f"model_{epoch}"
            model_name += f"_train_{avg_model_test_loss}"
            model_name += f"_test_{avg_model_test_loss}"
            # replace "." with "_" to avoid problems with file names
            model_name = model_name.replace(".", "_")
                # save the model
            torch.save(self.__model.state_dict(),
                        f"pre_training/{model_name}.pt")
            # save optimizer
            torch.save(self.__optimizer.state_dict(),
                        f"pre_training/optimizer_{epoch}.pt")

            # save the loss plot
            self.save_loss_plot(train_loss_history[epoch], epoch, "train")
            self.save_loss_plot(test_loss_history[epoch], epoch, "test")

            # decay the learning rate max_lr
            if epoch >= self.__training_args['lr_decay_start_epoch']:
                lr_upper = self.__training_args['lr'] - \
                    self.__training_args['lr_decay_step']
                lr_lower = self.__training_args['target_lr']
                if lr_upper < lr_lower:
                    lr_upper = lr_lower
                self.__optimizer.param_groups[0]['lr'] = lr_upper
                tqdm.tqdm.write(f"Lr bounds: {lr_lower} - {lr_upper}")
        conn.close()

    def test_value_loss(self) -> None:
        conn = sqlite3.connect(self.__training_args['db_path'])
        db_size = 0
        try:
            db_size = self.__training_args['db_size']
        except:
            db_size = conn.execute(
                "SELECT COUNT(*) FROM positions").fetchone()[0]
        batch_size = self.__training_args['batch_size']
        db_request_size = self.__training_args['db_request_size']
        num_requests = db_size // db_request_size
        num_batches = db_request_size // batch_size
        max_batches = self.__training_args['max_batches_per_epoch']
        idx1, idx2 = self.get_train_test_idx(num_requests, db_request_size)
        idx = np.concatenate((idx1, idx2))
        idx2 = idx

        models = os.listdir("pre_training")
        models = [model for model in models if model.endswith(".pt") and model.startswith("model")]

        for model in models:
            self.__model.load_state_dict(torch.load(f"pre_training/{model}"))
            self.__model.eval()
            avg_model_test_loss = 0

            pbar = tqdm.tqdm(total=len(idx2),
                             desc="Model " + str(model) + ". Testing.")
            self.__model.train()
            test_loss = 0
            pbar.reset()
            self.__model.eval()
            with torch.no_grad():
                for batch_num, test_idx in enumerate(idx2):
                    if batch_num >= self.__training_args["max_batches_per_epoch"]:
                        break
                    # get batch from the list
                    batch = self.get_pre_train_batch(
                        conn, batch_size, test_idx)
                    if batch is None:
                        continue
                    test_dataset = self.create_pre_training_dataset(
                        batch, batch_size)
                    for state, value in test_dataset:
                        state = self.__model.to_self(state)
                        value = self.__model.to_self(value)
                        policy_pred, value_pred = self.__model(state)
                        value_loss = self.__value_loss_fn(value_pred, value)
                        test_loss += value_loss.item()
                    test_loss /= num_batches
                    avg_model_test_loss += test_loss
                    pbar.update(1)
            avg_model_test_loss /= len(idx2)
            tqdm.tqdm.write(
                f"Model {model} - Test Loss: {avg_model_test_loss}")

    def load_best_model(self):
        models = os.listdir("pre_training")
        models = [model for model in models if model.endswith(".pt")]
        best_model = None
        best_model_loss = np.inf
        for model in models:
            if not model.startswith("model"):
                continue
            loss = model.split("_")[-1].replace("_", ".")
            # remove ".pt" from the end and convert to float
            loss = "0." + loss[:-3]
            loss = float(loss)
            if loss < best_model_loss:
                best_model_loss = loss
                best_model = model
        best_model_num = int(best_model.split("_")[1])
        self.__model.load_state_dict(torch.load(
            f"pre_training/{best_model}"))
        self.__model = self.__model.to(self.__model.device)
        # self.__optimizer.load_state_dict(torch.load(
        #     f"pre_training/optimizer_{best_model_num}.pt"))


if __name__ == "__main__":
    dml = torch_directml.device()
    # az = AlphaZero(device=dml)
    az = AlphaZero()
    # az.load_best_model()
    az.test_value_loss()
    # az.pre_train_value()
    # az.pre_train()
    # az.train()
