import os
import sqlite3
import torch
import yaml

import numpy as np
import torch_directml as dml
from data_loading.dataloader import DataLoader
from resnet.resnet import ResNet
from actionspace.actionspace import ActionSpace
from training.trainer import NetworkTrainer
from helper.helper_functions import *


def convert_to_numpy(arr):
    return np.frombuffer(arr, dtype=np.float32).reshape(8, 8, 8)


class TrainWrapper():
    def __init__(self, device=dml.device()) -> None:
        self.__load_args()
        self.__device = device
        self.__action_space = ActionSpace()

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
        model_attrs = {
            'num_blocks': self.__training_args['num_blocks'],
            'num_features': self.__training_args['num_features'],
            'num_input_features': self.__training_args['input_features'],
            'dtype': torch.float32,
            'squeeze_and_excitation': self.__training_args['squeeze_and_excitation'],
            'disable_policy': True,
            'policy_size': self.__action_space.size
        }
        resume_training = True
        db_size = self.get_db_size()
        b_size = int(512*8)
        slice_size = int(512)
        dataloader = DataLoader(
            db_path=self.__training_args['db_path'],
            table_name='positions',
            num_batches=200,
            batch_size=b_size,
            min_index=1,
            max_index=db_size,
            random=True,
            replace=False,
            shuffle=False,
            slice_size=slice_size,
            output_columns=['fen', 'cp'],
            to_tensor=True,
            specials={
                'fen': decode_from_fen,
                'cp': cp_to_value_clip
            },
            use_offsets=False
        )
        dataloader.start()
        val_dataloader = DataLoader(
            db_path=self.__training_args['db_path'],
            table_name='positions',
            num_batches=100,
            batch_size=b_size,
            min_index=db_size*0.95,
            max_index=db_size,
            random=False,
            replace=False,
            shuffle=False,
            slice_size=slice_size,
            output_columns=['fen', 'cp'],
            to_tensor=True,
            specials={
                'fen': decode_from_fen,
                'cp': cp_to_value_clip
            },
            use_offsets=False
        )
        val_dataloader.start()
        model = ResNet(
            num_blocks=self.__training_args['num_blocks'],
            num_features=self.__training_args['num_features'],
            num_input_features=self.__training_args['input_features'],
            squeeze_and_excitation=self.__training_args['squeeze_and_excitation'],
            policy_size=self.__action_space.size,
            device=self.__device,
            # weight_init_mode='xavier'
        )
        model.disable_policy()
        model.to(self.__device)
        trainer = NetworkTrainer(model, True, self.__device)
        path = 'model_files/models_'
        path += str(self.__training_args['input_features'])
        path += "_"
        path += str(self.__training_args['num_blocks'])
        path += "_"
        path += str(self.__training_args['num_features'])
        if self.__training_args['squeeze_and_excitation']:
            path += "_se"
        path += "/"
        if not os.path.exists(path):
            os.makedirs(path)
            resume_training = False
        else:
            if len(os.listdir(path)) == 0:
                resume_training = False

        trainer.fit(dataloader, max_epochs=150, early_stopping=10, path=path, save_attrs=model_attrs,
                    resume=resume_training, resume_optimizer=False, log_plot=True)

    def get_db_size_old(self):
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
    
    def get_db_size(self):
        db_name = os.path.basename(self.__training_args['db_path'])
        if os.path.exists(f"data_loading/db_size.yaml"):
            with open(f"data_loading/db_size.yaml", "r") as db_size_file:
                db_size_dict = yaml.safe_load(db_size_file)
        else:
            db_size_dict = {}
        if db_name in db_size_dict:
            db_size = db_size_dict[db_name]
        else:
            conn = sqlite3.connect(self.__training_args['db_path'])
            db_size = conn.execute(
                "SELECT COUNT(*) FROM positions").fetchone()[0]
            conn.close()
            db_size_dict[db_name] = db_size
            with open(f"data_loading/db_size.yaml", "w") as db_size_file:
                yaml.dump(db_size_dict, db_size_file)
        return db_size

    def find_lr(self):
        db_size = self.get_db_size()
        dataloader = DataLoader(
            db_path=self.__training_args['db_path'],
            table_name='positions',
            num_batches=1000,
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
        model = ResNet(
            self.__training_args['num_blocks'],
            self.__training_args['num_features'],
            self.__training_args['input_features'],
            self.__action_space.size,
            se=self.__training_args['squeeze_and_excitation']
        )
        model.disable_policy()
        model.to(self.__device)
        trainer = NetworkTrainer(model, True, self.__device)
        trainer.lr_finder(dataloader, 1e-8, 1, 0.005, 100)
        # trainer.lr_finder(tmp, 1e-4, 0.1, 0.0015, 100)

    def find_params(self):
        sample_count = 100*512
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
            max_index=db_size*0.9,
            random=True,
            replace=False,
            shuffle=True,
            slice_size=16,
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
        trainer.optuna_study(dataloader, study_name, 100)

    def predict(self):
        db_size = self.get_db_size()
        model = ResNet(
            num_blocks=self.__training_args['num_blocks'],
            num_features=self.__training_args['num_features'],
            num_input_features=self.__training_args['input_features'],
            squeeze_and_excitation=self.__training_args['squeeze_and_excitation'],
            policy_size=self.__action_space.size,
            device=self.__device
        )
        checkpoint = torch.torch.load(
            "models_15_8_96/model_24_0.00393247862579301.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.disable_policy()
        model.to(self.__device)
        model.eval()
        with torch.no_grad():
            while True:
                conn = sqlite3.connect(self.__training_args['db_path'])
                cur = conn.cursor()
                idx = np.random.randint(1, 1000)
                db_slice = cur.execute(
                    "SELECT * FROM positions LIMIT 1 OFFSET {}".format(idx)).fetchall()
                conn.close()
                id, enc, fen, cp, policy, value, samples, legal_moves = db_slice[0]
                value = cp_to_value_clip(cp)
                m_in = decode_from_fen(fen)
                m_in = torch.tensor(m_in).unsqueeze(0)
                m_in = m_in.to(self.__device)
                v_hat = model(m_in)
                v_hat = v_hat.cpu().numpy()
                print(fen, value, v_hat)
                print()

    def load_and_print_params(self):
        path = 'models/model_1_0.0853324470296502_simple.pth'
        model = ResNet(
            self.__args['num_blocks'],
            self.__args['num_features'],
            self.__args['input_features'],
            self.__action_space.size
        )
        model.disable_policy()
        checkpoint = torch.torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.ASGD(
            model.parameters(),
            lr=1e-2,
            weight_decay=1e-2,
            # alpha=0.5,
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        params = optimizer.state_dict()['param_groups'][0]
        for param in params:
            print(param)

    def train_all(self):
        model_attrs = {
            'num_blocks': self.__training_args['num_blocks'],
            'num_features': self.__training_args['num_features'],
            'num_input_features': self.__training_args['input_features'],
            'dtype': torch.float32,
            'squeeze_and_excitation': self.__training_args['squeeze_and_excitation'],
            'disable_policy': False,
            'policy_size': self.__action_space.size
        }
        resume_training = True
        db_size = self.get_db_size()
        b_size = int(512*8)
        slice_size = int(b_size)
        dataloader = DataLoader(
            db_path=self.__training_args['db_path'],
            table_name='positions',
            num_batches=100,
            batch_size=b_size,
            min_index=1,
            max_index=db_size*0.95,
            random=True,
            replace=False,
            shuffle=False,
            slice_size=slice_size,
            output_columns=['fen', 'cp', 'policy'],
            to_tensor=True,
            specials={
                'fen': decode_from_fen,
                'cp': cp_to_value_clip,
                'policy': decode_policy
            },
            use_offsets=False
        )
        dataloader.start()
        val_dataloader = DataLoader(
            db_path=self.__training_args['db_path'],
            table_name='positions',
            num_batches=50,
            batch_size=b_size,
            min_index=db_size*0.95,
            max_index=db_size,
            random=False,
            replace=False,
            shuffle=False,
            slice_size=slice_size,
            output_columns=['fen', 'cp', 'policy'],
            to_tensor=True,
            specials={
                'fen': decode_from_fen,
                'cp': cp_to_value_clip,
                'policy': decode_policy
            },
            use_offsets=False
        )
        val_dataloader.start()
        model = ResNet(
            num_blocks=self.__training_args['num_blocks'],
            num_features=self.__training_args['num_features'],
            num_input_features=self.__training_args['input_features'],
            squeeze_and_excitation=self.__training_args['squeeze_and_excitation'],
            policy_size=self.__action_space.size,
            device=self.__device,
            # weight_init_mode='xavier'
        )
        model.to(self.__device)
        trainer = NetworkTrainer(model, False, self.__device)
        path = 'models_'
        path += str(self.__training_args['input_features'])
        path += "_"
        path += str(self.__training_args['num_blocks'])
        path += "_"
        path += str(self.__training_args['num_features'])
        if self.__training_args['squeeze_and_excitation']:
            path += "_se"
        path += "/"
        if not os.path.exists(path):
            os.makedirs(path)
            resume_training = False
        else:
            if len(os.listdir(path)) == 0:
                resume_training = False

        trainer.fit(dataloader, val_dataloader, 100, 5, path=path, save_attrs=model_attrs,
                    resume=resume_training, resume_optimizer=False, log_plot=True)


if __name__ == "__main__":
    az = TrainWrapper()
    # az.find_lr()
    # az.load_and_print_params()
    # az.find_params()
    az.train_value()
    # az.predict()
    # az.train_all()
