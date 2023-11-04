import os
import sqlite3
import torch
import yaml
import tqdm

import numpy as np
import torch_directml as dml
from cache_read_priority import Cache
from db_dataloader import DataLoader
from mcts import MCTS
from resnet import ResNet
from actionspace import ActionSpace
import sys
from trainer import NetworkTrainer


def convert_to_numpy(arr):
    return np.frombuffer(arr, dtype=np.float32).reshape(8, 8, 8)


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
        db_size = self.get_db_size()
        dataloader = DataLoader(
            db_path=self.__training_args['db_path'],
            table_name='positions',
            num_batches=0,
            batch_size=512,
            min_index=1,
            max_index=db_size*0.9,
            random=True,
            replace=True,
            shuffle=True,
            slice_size=64,
            data_cols=['encoded'],
            label_cols=['prob'],
            to_tensor=True,
            specials={'encoded': convert_to_numpy}
        )
        dataloader.start()
        val_dataloader = DataLoader(
            db_path=self.__training_args['db_path'],
            table_name='positions',
            num_batches=0,
            batch_size=512,
            min_index=db_size*0.9,
            max_index=db_size,
            random=False,
            replace=False,
            shuffle=False,
            slice_size=64,
            data_cols=['encoded'],
            label_cols=['prob'],
            to_tensor=True,
            specials={'encoded': convert_to_numpy}
        )
        val_dataloader.start()
        model = ResNet(
            self.__args['num_blocks'],
            self.__args['num_features'],
            self.__args['input_features'],
            self.__action_space.size,
            se=False
        )
        model.disable_policy()  
        model.to(self.__device)
        trainer = NetworkTrainer(model, True, self.__device)
        trainer.fit(dataloader, val_dataloader, 15, 5, path="models/", plot=True)
        # trainer.fit(dataloader, 1, 5, path="models/", plot=True)

    def get_db_size(self):
        db_name = os.path.basename(self.__training_args['db_path'])
        if os.path.exists(f"db_size_{db_name}.txt"):
            with open(f"db_size_{db_name}.txt", "r") as db_size_file:
                db_size = int(db_size_file.read())
        else:
            conn = sqlite3.connect(self.__training_args['db_path'])
            db_size = conn.execute(
                "SELECT COUNT(*) FROM positions").fetchone()[0]
            conn.close()
            # save db size to file
            db_name = os.path.basename(self.__training_args['db_path'])
            with open(f"db_size_{db_name}.txt", "w") as db_size_file:
                db_size_file.write(str(db_size))
        return db_size

    def find_lr(self):
        db_size = self.get_db_size()
        tmp = DataLoader(
            db_path=self.__training_args['db_path'],
            table_name='positions',
            num_batches=0,
            batch_size=1024,
            min_index=1,
            max_index=db_size,
            random=True,
            replace=False,
            shuffle=True,
            slice_size=128,
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
        trainer.lr_finder(tmp, 1e-8, 1e-1, 0.005, 50)
        # trainer.lr_finder(tmp, 1e-4, 0.1, 0.0015, 100)

    def find_params(self):
        sample_count = 100*1024
        batch_size = 512
        num_batches = sample_count//batch_size
        db_size = self.get_db_size()
        squeeze_and_excitation = False
        dataloader = DataLoader(
            db_path=self.__training_args['db_path'],
            table_name='positions',
            num_batches=num_batches,
            batch_size=batch_size,
            min_index=1,
            max_index=db_size,
            random=False,
            replace=False,
            shuffle=False,
            slice_size=64,
            specials={'encoded': convert_to_numpy},
            data_cols=['encoded'],
            label_cols=['prob']
        )
        dataloader.start()
        model = ResNet(
            self.__args['num_blocks'],
            self.__args['num_features'],
            self.__args['input_features'],
            self.__action_space.size,
            se=squeeze_and_excitation
        )
        model.disable_policy()
        model.to(self.__device)
        trainer = NetworkTrainer(model, True, self.__device)
        study_name = f"optuna_study_{self.__args['num_blocks']}_{self.__args['num_features']}"
        if squeeze_and_excitation:
            study_name += "_se"
        trainer.optuna_study(dataloader, study_name, 75)

    def predict(self):
        db_size = self.get_db_size()
        model = ResNet(
            self.__args['num_blocks'],
            self.__args['num_features'],
            self.__args['input_features'],
            self.__action_space.size
        )
        model.disable_policy()
        checkpoint = torch.torch.load(
            "models12_144/model_15_0.4218898293375969.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.__device)
        model.eval()
        with torch.no_grad():
            while True:
                conn = sqlite3.connect(self.__training_args['db_path'])
                cur = conn.cursor()
                cur.execute(
                    "SELECT encoded, fen, prob FROM positions WHERE id=?", (np.random.randint(1, db_size),))
                batch = cur.fetchone()
                conn.close()
                enc, fen, prob = batch
                enc = convert_to_numpy(enc)
                enc = torch.from_numpy(enc).unsqueeze(0)
                enc = enc.to(self.__device)
                prob_pred = model(enc)
                prob_pred = prob_pred.cpu().numpy()
                print(fen, prob, prob_pred)
                print()


if __name__ == "__main__":
    az = AlphaZero()
    # az.find_params()
    az.train_value()
    # az.predict()
