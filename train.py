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
          early_stopping=False, precision="bf16-mixed"):
    logger = TensorBoardLogger("lightning_logs", name=logger_name)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    callbacks = [checkpoint_callback]
    if early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)

    trainer = Trainer(max_epochs=max_epochs, logger=logger,
                      callbacks=callbacks, precision=precision)
    trainer.fit(litmodel, train_loader, val_loader)

     # save best model
    litmodel = LitResNet.load_from_checkpoint(checkpoint_callback.best_model_path, model=litmodel.model)
    torch.save({
        'model_state_dict': litmodel.model.state_dict(),
        'num_blocks': config['num_blocks'],
        'num_features': config['num_features'],
        'num_input_features': config['num_input_features'],
    }, 'model_files/model.pth')


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
    batch_size = 4096
    slice_size = 256
    train_df = pd.read_csv('data/train.csv')
    train_df = train_df[['fen', 'win_prob']]
    train_data = PandasDataset(dataframe=train_df)
    sampler = SliceSampler(train_data, slice_size, batch_size)
    train_loader = DataLoader(
        train_data,
        batch_sampler=sampler,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_df = pd.read_csv('data/test.csv')
    test_df = test_df[['fen', 'win_prob']]
    test_data = PandasDataset(dataframe=test_df)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, test_loader


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('medium')

    # Create dataloaders
    train_loader, test_loader = create_dataloaders()

    # Load config
    with open('train_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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
          early_stopping=True, max_epochs=3)
