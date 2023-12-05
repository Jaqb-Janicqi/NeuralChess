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
from lit_resnet import LitResNet


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

    def lit_train_value(self):
        conn = sqlite3.connect(self.__training_args['db_path'])
        db_size = conn.execute(
                "SELECT COUNT(*) FROM positions").fetchone()[0]
        conn.close()
        dataloader = DataLoader(
            db_path='C:/sqlite_chess_db/chess_positions.db',
            table_name='positions',
            num_batches=0,
            batch_size=1024,
            min_index=1,
            max_index=db_size,
            random=True,
            replace=False,
            shuffle=True,
            slice_size=64,
            specials={'encoded': convert_to_numpy}
        )
        dataloader.start()
        model = ResNet(
            self.__args['num_blocks'],
            self.__args['num_features'],
            self.__args['input_features'],
            self.__action_space.size
        )
        model.to(self.__device)
        # model(torch.rand((1, 16, 8, 8)).to(dml.device()))
        lit_model = LitResNet(model, True, self.__device)
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=100,
            callbacks=[pl.callbacks.LearningRateMonitor()],
            log_every_n_steps=1,
            fast_dev_run=True,
            reload_dataloaders_every_n_epochs=1,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        trainer.fit(lit_model, dataloader)

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
        num_requests = self.__training_args['num_requests_pre']
        num_batches = db_request_size // batch_size
        batches_per_request = db_request_size // batch_size
        indexes = np.arange(num_requests)
        np.random.shuffle(indexes)
        num_epochs = 1

        self.__optimizer = torch.optim.AdamW(
            self.__model.parameters(),
            lr=1e-3,
            weight_decay=0.1,
            betas=(0.85, 0.95),
        )
        self.__scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.__optimizer,
            max_lr=1e-3,
            total_steps=num_requests * batches_per_request,
            three_phase=True,
            cycle_momentum=True,
            base_momentum=0.1,
            max_momentum=0.9,
        )

        # pre-train the value head
        pbar = tqdm.tqdm(
            total=num_requests,
            desc=""
        )
        loss_history = []
        loss_stop_window = 10
        loss_window = []
        total_loss = 0
        batch_num = 0
        last_lr = 0
        early_stop = False
        for epoch in range(num_epochs):
            if early_stop:
                break
            self.__model.train()
            for request_num, db_index in enumerate(indexes):
                if early_stop:
                    break
                dataset = self.get_random_batch(
                    db_request_size, batch_size, db_index)
                if dataset is None:
                    continue

                last_loss = 0
                for state, value in dataset:
                    state = self.__model.to_self(state)
                    value = self.__model.to_self(value)
                    policy_pred, value_pred = self.__model(state)
                    value_loss = self.__value_loss_fn(value_pred, value)
                    self.__optimizer.zero_grad()
                    value_loss.backward()
                    self.__optimizer.step()
                    self.__scheduler.step()

                    total_loss += value_loss.item()
                    last_loss += value_loss.item()
                    loss_history.append(value_loss.item())
                    batch_num += 1

                # update progress bar
                pbar.update(1)
                desc = f"Epoch {epoch}."
                desc += f" Lr: {self.__scheduler.get_last_lr()}"
                desc += f" Avg Loss: {total_loss / (batch_num + 1)}"
                desc += f" Loss: {last_loss / batches_per_request}"
                pbar.set_description(desc)

                if request_num % (num_requests // 10) == 0 and request_num != 0:
                    self.save_loss_plot(loss_history, epoch,
                                        "episode " + str(num_requests // 10))
                    # save the model
                    model_name = f"model_{epoch}"
                    model_name += f"_blocks_{self.__args['num_blocks']}"
                    model_name += f"_features_{self.__args['num_features']}"
                    model_name += f"_batches_{batch_num}"
                    model_name += f"_reqsize_{db_request_size}"
                    model_name += f"_loss{np.sum(np.array(loss_window)) / len(loss_window)}"
                    # replace "." with "_" to avoid problems with file names
                    model_name = model_name.replace(".", "_")
                    # save the model
                    torch.save(
                        self.__model.state_dict(),
                        f"pre_training/{model_name}.pt"
                    )
                    # save optimizer
                    optimizer_name = f"optimizer_{epoch}"
                    optimizer_name += f"_blocks_{self.__args['num_blocks']}"
                    optimizer_name += f"_features_{self.__args['num_features']}"
                    torch.save(
                        self.__optimizer.state_dict(),
                        f"pre_training/{optimizer_name}.pt"
                    )

                    # save the loss plot
                    self.save_loss_plot(loss_history, epoch, "train")
                    conn.close()

                # if the learning rate is still increasing, continue training
                if self.__scheduler.get_last_lr()[0] > last_lr:
                    last_lr = self.__scheduler.get_last_lr()[0]
                    continue

                loss_window.append(last_loss / batches_per_request)
                if len(loss_window) > loss_stop_window + 1:
                    loss_window = loss_window[-loss_stop_window:]

                # check if the loss is not decreasing
                if len(loss_window) == loss_stop_window:
                    if np.all(np.array(loss_window) <= loss_window[-1]):
                        tqdm.tqdm.write(
                            f"Loss not decreasing. Stopping training.")
                        early_stop = True
                        break

        # save the model
        model_name = f"model_{epoch}"
        model_name += f"_blocks_{self.__args['num_blocks']}"
        model_name += f"_features_{self.__args['num_features']}"
        model_name += f"_batches_{batch_num}"
        model_name += f"_reqsize_{db_request_size}"
        model_name += f"_loss{np.sum(np.array(loss_window)) / len(loss_window)}"
        # replace "." with "_" to avoid problems with file names
        model_name = model_name.replace(".", "_")
        # save the model
        torch.save(
            self.__model.state_dict(),
            f"pre_training/{model_name}.pt"
        )
        # save optimizer
        optimizer_name = f"optimizer_{epoch}"
        optimizer_name += f"_blocks_{self.__args['num_blocks']}"
        optimizer_name += f"_features_{self.__args['num_features']}"
        torch.save(
            self.__optimizer.state_dict(),
            f"pre_training/{optimizer_name}.pt"
        )

        # save the loss plot
        self.save_loss_plot(loss_history, epoch, "train")
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
        models = [model for model in models if model.endswith(
            ".pt") and model.startswith("model")]

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

    def predict(self):
        from mcts_node import Node
        path = "pre_training/model_0_blocks_12_features_240_batches_99968_reqsize_8192_loss0_05953811692486977.pt"
        model_attributes = path.split("_")
        self.__args['num_blocks'] = int(model_attributes[4])
        self.__args['num_features'] = int(model_attributes[6])
        self.__model = ResNet(
            self.__args['num_blocks'],
            self.__args['num_features'],
            self.__args['input_features'],
            self.__action_space.size,
            self.__model.device
        )
        self.__model.load_state_dict(torch.load(path))
        self.__model.eval()
        conn = sqlite3.connect(self.__training_args['db_path'])
        if self.__training_args['db_size'] is None:
            db_size = conn.execute(
                "SELECT COUNT(*) FROM positions").fetchone()[0]
        else:
            db_size = self.__training_args['db_size']
        while True:
            random_idx = np.random.randint(0, db_size)
            batch = self.get_pre_train_batch(conn, 1, random_idx)
            if batch is None:
                continue
            db_idx, fen, cp, prob = batch[0]
            state = Node(0, chess.Board(fen), self.__action_space)
            with torch.no_grad():
                tensor_state = self.__model.get_unbatched_tensor_state(
                    state.encoded)
                policy, value = self.__model(tensor_state)
                value = self.__model.get_value(value)
                print(f"Fen: {fen} - Value: {value} - CP: {cp} - Prob: {prob}")
                print()


if __name__ == "__main__":
    az = AlphaZero()
    # az.predict()
    # az.pre_train_value()
    # az.test_value_loss()
    az.lit_train_value()
