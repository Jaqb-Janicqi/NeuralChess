import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from resnet.litresnet import LitResNet
from resnet.resnet import ResNet
import yaml
from torch.utils.data import DataLoader
from data_loading.dataset import PandasDataset
from data_loading.sampler import SliceSampler
from helper.helper_functions import decode_from_fen


def train(litmodel, train_loader, val_loader, max_epochs=100, logger_name='default',
          early_stopping=0, precision="bf16-mixed"):
    logger = TensorBoardLogger("lightning_logs", name=logger_name)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='model-{epoch:03d}-{val_loss:.6f}',
        save_top_k=5,
        mode='min',
    )
    callbacks = [checkpoint_callback]
    if early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping,
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)

    trainer = Trainer(max_epochs=max_epochs, logger=logger,
                      callbacks=callbacks, precision=precision)
    trainer.fit(litmodel, train_loader, val_loader)

    # Extract validation loss from the best model checkpoint filename
    best_val_loss = float(checkpoint_callback.best_model_path.split(
        '=')[-1][:-5])  # Extracting the validation loss
    # save best model
    litmodel = LitResNet.load_from_checkpoint(
        checkpoint_callback.best_model_path, model=litmodel.model)
    torch.save({
        'model_state_dict': litmodel.model.state_dict(),
        'num_blocks': config['num_blocks'],
        'num_features': config['num_features'],
        'num_input_features': config['num_input_features'],
    }, f'model_files/model_{best_val_loss:.6f}.pth')


def collate_fn(batch):
    data, target = zip(*batch)

    tmp = []
    for d in data:
        tmp.append(torch.tensor(decode_from_fen(d), dtype=torch.float32))
    data = tmp

    data = torch.stack(data)
    target = torch.tensor(target, dtype=torch.float32)
    return data, target


def create_dataloaders():
    batch_size = 2048
    slice_size = 512
    num_workers = 6

    stockfish_df = pd.read_csv('data/stockfish_200ms.csv')
    stockfish_df = stockfish_df[['fen', 'win_prob']]
    stockfish_df_test = stockfish_df.sample(frac=0.05)
    stockfish_df_train = stockfish_df.drop(stockfish_df_test.index)

    train_df = pd.read_csv('data/train.csv')
    train_df = train_df[['fen', 'win_prob']]
    train_df = pd.concat([train_df, stockfish_df_train])
    train_data = PandasDataset(dataframe=train_df)
    sampler = SliceSampler(train_data, slice_size, batch_size)
    train_loader = DataLoader(
        train_data,
        batch_sampler=sampler,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_df = pd.read_csv('data/test.csv')
    test_df = test_df[['fen', 'win_prob']]
    test_df = pd.concat([test_df, stockfish_df_test])
    test_data = PandasDataset(dataframe=test_df)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, test_loader


if __name__ == '__main__':
    # torch.manual_seed(0)
    torch.set_float32_matmul_precision('high')

    # Create dataloaders
    train_loader, test_loader = create_dataloaders()

    config = {
        'num_blocks': 6,
        'num_features': 64,
        'num_input_features': 15,
        'weight_init_mode': "kaiming"
    }

    # Create model
    model = ResNet(
        num_blocks=config['num_blocks'],
        num_features=config['num_features'],
        num_input_features=config['num_input_features'],
        weight_init_mode=config['weight_init_mode']
    ).cuda()
    litmodel = LitResNet(model)

    # Train model
    train(litmodel, train_loader, test_loader,
          early_stopping=5, max_epochs=50)
