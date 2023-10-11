import copy
import os
import pickle
import sqlite3
import time
import matplotlib.pyplot as plt
import yaml
import tqdm
import random

import numpy as np
import torch
import torch_directml
import chess
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from cache_write_priority import Cache
from mcts import MCTS
from resnet import ResNet
from actionspace import ActionSpace
from state import State


class AlphaZero():
    def __init__(self, device="cpu") -> None:
        self.__load_args()
        self.__action_space = ActionSpace()
        self.__board = chess.Board()
        self.__state = State(self.__board)
        self.__model = ResNet(
            self.__args['num_blocks'],
            self.__args['num_features'],
            self.__args['input_features'],
            self.__action_space.size,
            device
        )
        # self.__optimizer = torch.optim.AdamW(
        #     self.__model.parameters(),
        #     lr=self.__training_args['lr'],
        #     weight_decay=self.__training_args['weight_decay']
        # ) # Avg Train Loss: 0.17078006560685205
        # self.__optimizer = torch.optim.SGD(
        #     self.__model.parameters(),
        #     lr=self.__training_args['lr'],
        #     momentum=self.__training_args['momentum'],
        #     weight_decay=self.__training_args['weight_decay']
        # )   # Avg Train Loss: 0.05852350499480963
        self.__optimizer = torch.optim.ASGD(
            self.__model.parameters(),
            lr=self.__training_args['lr'],
            weight_decay=self.__training_args['weight_decay']
        )
        self.__model_num = 1
        self.__value_loss_fn = nn.MSELoss()
        self.__policy_loss_fn = nn.CrossEntropyLoss()
        self.__evaluation_cache = Cache(self.__training_args['cache_size'])

    def __load_args(self):
        with open("config.yaml", "r") as config_file:
            self.__args = yaml.safe_load(config_file)
        with open("training_config.yaml", "r") as training_config_file:
            self.__training_args = yaml.safe_load(training_config_file)

    def step(self, probs):
        action = self.__action_space[np.argmax(probs)]
        self.__state = self.__state.next_state(action)

    def stochastic_step(self, probs):
        t = 1 if "pit_temp" not in self.__args else self.__args["pit_temp"]
        probs **= (1/t)
        probs = probs / np.sum(probs)
        action_id = np.random.choice(np.arange(len(probs)), p=probs)
        action = self.__action_space[action_id]
        self.__state = self.__state.next_state(action)

    def pit(self, num_games, model1, model2, stochastic=False, verbose=False) -> float:
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

    def execute_epoch(self, dataloader, model):
        for data in dataloader:
            state, policy, value = data
            state = state.to(self.__model.device).unsqueeze(1).float()
            policy = policy.to(self.__model.device).unsqueeze(1).float()
            value = value.to(self.__model.device).float()
            policy_pred, value_pred = model(state)
            policy_pred = policy_pred.unsqueeze(0)
            policy = policy.squeeze(1).unsqueeze(0)
            value_pred = value_pred.squeeze(1)

            value_loss = self.__value_loss_fn(value_pred, value)
            policy_loss = self.__value_loss_fn(policy_pred, policy)
            loss = value_loss + policy_loss
            if loss.item() is np.nan:
                return None
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
        return model

    def saturate(self, dataloader):
        self.__optimizer.param_groups[0]['lr'] = self.args['saturation_lr']
        new_model = copy.deepcopy(self.model)
        for i in trange(self.args['saturation_patience'], desc="Saturating"):
            new_model.train()
            for epoch in range(self.args['saturation_epochs']):
                temp = copy.deepcopy(new_model)
                temp = self.execute_epoch(
                    dataloader, temp, self.args['saturation_lr'])
                if temp is not None:
                    new_model = temp
            new_model.eval()
            with torch.no_grad():
                winrate = self.pit(100, self.model, new_model)
            if winrate > 0.55:
                torch.save(new_model.state_dict(),
                           f"model_{self.model_num}.pt")
                self.model = new_model
                i = 0
            self.model_num += 1
            with open("winratio.txt", "a") as f:
                f.write(f"saturation: {winrate}\n")

    def train(self, load_dataset=False, extend_dataset=True, only_last_model=False):
        self.optimizer.param_groups[0]['lr'] = self.args['lr']
        self.model.eval()
        # reset winratio file
        with open("winratio.txt", "w") as f:
            f.write("")
        if load_dataset:
            with open("dataset.pkl", "rb") as f:
                self.dataset = pickle.load(f)
        for episode in range(self.args['num_episodes']):
            self.model.eval()
            if extend_dataset:
                with torch.no_grad():
                    data = []
                    for _ in trange(self.__training_args['num_games'], desc="Self Play"):
                        data.extend(self.game.self_play(
                            self.model, self.evaluation_cache, True))
                    self.dataset.extend(data)
                with open("dataset.pkl", "wb") as f:
                    pickle.dump(self.dataset, f)
            elif self.dataset == []:
                with open("dataset.pkl", "rb") as f:
                    self.dataset = pickle.load(f)
            dataloader = DataLoader(
                self.dataset, batch_size=self.args['batch_size'], shuffle=True)
            new_model = copy.deepcopy(self.model)
            new_model.train()
            for epoch in range(self.args['num_epochs']):
                temp = copy.deepcopy(new_model)
                temp = self.execute_epoch(dataloader, temp)
                if temp is not None:
                    new_model = temp
            self.__evaluation_cache.clear()

            new_model.eval()
            with torch.no_grad():
                winrate = self.pit(100, self.model, new_model, True)
            if winrate > 0.55:
                if only_last_model:
                    self.dataset = []
                # append winratio to file
                with open("winratio.txt", "a") as f:
                    f.write(f"{winrate}\n")
                torch.save(new_model.state_dict(),
                           f"model_{self.model_num}.pt")
                self.model = new_model
            self.model_num += 1
        self.saturate(dataloader)

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

    def get_train_test_idx(self, num_batches, batch_size):
        training_split = self.__training_args['training_split']
        if training_split is None:
            training_split = 0.9
        training_split = int(num_batches * training_split)
        idx = np.arange(num_batches)
        train_idx = idx[:training_split]
        test_idx = idx[training_split:]
        return train_idx, test_idx

    def pre_train_value(self):
        # self.__model.load_state_dict(torch.load("C:/Users/janic/Desktop/04676_06569.pt"))

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
        # self.save_loss_plot(np.arange(100), 0, "train")

        # pre-train the value head
        num_epochs = self.__training_args['num_epochs_pre']
        train_loss_history = [[] for _ in range(num_epochs)]
        test_loss_history = [[] for _ in range(num_epochs)]
        for epoch in range(num_epochs):
            # shuffle indexes to rotate dataset if max_batches is set
            np.random.shuffle(train_idx_list)
            np.random.shuffle(test_idx_list)
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
            self.__model.train()
            for batch_num, train_idx in enumerate(train_idx_list):
                if batch_num >= max_batches:
                    train_batches_num = batch_num
                    break
                # get a random batch
                batch = self.get_pre_train_batch(conn, batch_size, train_idx)
                if batch is None:
                    continue
                dataset = self.create_pre_training_dataset(batch, batch_size)

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
            avg_model_test_loss /= len(test_idx_list)
            tqdm.tqdm.write(
                f"Epoch {epoch} - Train Loss: {avg_model_train_loss} - Test Loss: {avg_model_test_loss}")
            # print(
            #     f"Epoch {epoch} - Train Loss: {train_loss} - Test Loss: {test_loss}")

            # save the best model
            if avg_model_test_loss < best_model_loss:
                best_model_loss = avg_model_test_loss
                model_name = f"pre_training/model_{epoch}_"
                model_name += f"{avg_model_test_loss}"
                # replace "." with "_" to avoid problems with file names
                model_name = model_name.replace(".", "_")
                model_name += ".pt"
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


if __name__ == "__main__":
    dml = torch_directml.device()
    az = AlphaZero(device=dml)
    az.pre_train_value()
