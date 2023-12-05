import os
import sqlite3
import torch
import torch_directml as dml
import yaml

from db_dataloader import DataLoader
from chess_model import ChessModel
from helper_functions import *
from trainer import NetworkTrainer


class EngineTrainer():
    def __init__(self, device=dml.device()) -> None:
        self.__load_args()
        self.__device = device

    def __load_args(self):
        with open("robohub_training_config.yaml", "r") as training_config_file:
            self.__training_args = yaml.safe_load(training_config_file)

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

    def train_value(self):
        model_attrs = {
            'num_blocks': self.__training_args['num_blocks'],
            'num_features': self.__training_args['num_features'],
            'num_input_features': self.__training_args['input_features'],
            'num_dense': self.__training_args['num_dense'],
            'dense_size': self.__training_args['dense_size'],
            'dtype': torch.float32,
            'squeeze_and_excitation': self.__training_args['squeeze_and_excitation'],
            'dense_reduce_factor': self.__training_args['dense_reduce_factor']
        }
        resume_training = True
        db_size = self.get_db_size()
        b_size = int(512*8)
        slice_size = int(b_size)
        dataloader = DataLoader(
            db_path=self.__training_args['db_path'],
            table_name='positions',
            num_batches=200, 
            batch_size=b_size,
            min_index=1,
            max_index=db_size*0.95,
            random=True,
            replace=False,
            shuffle=False,
            slice_size=slice_size,
            output_columns=['fen', 'cp'],
            to_tensor=True,
            specials={'fen': decode_from_fen_bit},
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
            output_columns=['fen', 'cp'],
            to_tensor=True,
            specials={'fen': decode_from_fen_bit},
            use_offsets=False
        )
        val_dataloader.start()
        model = ChessModel(
            num_blocks=self.__training_args['num_blocks'],
            num_features=self.__training_args['num_features'],
            num_input_features=self.__training_args['input_features'],
            num_dense=self.__training_args['num_dense'],
            dense_size=self.__training_args['dense_size'],
            dtype=torch.float32,
            squeeze_and_excitation=self.__training_args['squeeze_and_excitation'],
            dense_reduce_factor=self.__training_args['dense_reduce_factor']

        )
        model.to(self.__device)
        trainer = NetworkTrainer(model, True, self.__device)
        path = 'models_'
        path += str(self.__training_args['input_features'])
        path += "_"
        path += str(self.__training_args['num_blocks'])
        path += "_"
        path += str(self.__training_args['num_features'])
        path += "_"
        path += str(self.__training_args['num_dense'])
        path += "_"
        path += str(self.__training_args['dense_size'])
        if self.__training_args['squeeze_and_excitation']:
            path += "_se"
        path += "/"
        if not os.path.exists(path):
            os.makedirs(path)
            resume_training = False
        else:
            if len(os.listdir(path)) == 0:
                resume_training = False

        trainer.fit(dataloader, val_dataloader, 20, 5, path=path, save_attrs=model_attrs,
                    resume=resume_training, resume_optimizer=False, log_plot=True)
        
if __name__ == "__main__":
    et = EngineTrainer()
    et.train_value()