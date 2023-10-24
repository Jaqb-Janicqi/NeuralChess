import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml as dml
from tqdm import tqdm
from scipy.signal import savgol_filter
import pickle

class TrainingModule:
    def __init__(self, model, device=dml.device()):
        self._model: nn.Module = model
        self._device = device
        self._optimizers = {}  # {key: optimizer}
        self._log_dict = {}  # {key: [value]}
        self._step_list = []  # [class]
        self._dataloaders = {}  # {key: dataloader}

    def log(self, key, value):
        if key not in self._log_dict:
            self._log_dict[key] = []
        self._log_dict[key].append(value)

    def training_step(self, batch):
        x, y = batch
        x = torch.tensor(x).to(self._device)
        y = torch.tensor(y).to(self._device)
        y_hat = self._model(x)
        loss = F.cross_entropy(y_hat, y)
        del x, y, y_hat
        self.log('train_loss', loss.item())
        return loss

    def test_step(self, batch):
        x, y = batch
        x = torch.tensor(x).to(self._device)
        y = torch.tensor(y).to(self._device)
        y_hat = self._model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss.item())
        return loss

    def fit(self, train_loader, val_loader=None, max_epochs=None, early_stopping=False, save_path=None):
        self._dataloaders['train_loader'] = train_loader
        if val_loader is not None:
            self._dataloaders['val_loader'] = val_loader
        self.configure_optimizers()
        start_epoch = 0
        if save_path is not None:
            start_epoch, _ = self.load(save_path)
        if max_epochs is None:
            max_epochs = sys.maxsize ** 10
        for epoch in range(start_epoch, max_epochs):
            pbar = tqdm(
                train_loader, desc=f'Epoch {epoch}', leave=False, dynamic_ncols=True, total=len(train_loader))

            self._model.train()
            for batch in train_loader:
                loss = self.training_step(batch)
                self._optimizers['optimizer'].zero_grad()
                loss.backward()
                for func in self._step_list:
                    func.step()

            if val_loader is None:
                self.log('avg_loss',
                         sum(self._log_dict['train_loss']) / len(self._log_dict['train_loss']))
            else:
                self._model.eval()
                for batch in val_loader:
                    loss = self.test_step(batch)
                self.log('avg_loss',
                         sum(self._log_dict['val_loss']) / len(self._log_dict['val_loss']))

            if early_stopping:
                if len(self._log_dict['avg_loss']) >= 10:
                    if self._log_dict['avg_loss'][-1] > np.all(self._log_dict['avg_loss'][-10:-2]):
                        break

            self.save('models/', epoch, self._log_dict['avg_loss'][-1])
            self.on_epoch_end(self._log_dict)

    def save(self, path, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizers['optimizer'].state_dict(),
            'loss': loss,

        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizers['optimizer'].load_state_dict(
            checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

    def configure_optimizers(self):
        self._optimizers['optimizer'] = torch.optim.AdamW(
            self._model.parameters(),
        )
        self._step_list.append(self._optimizers['optimizer'])

    @property
    def steps_per_epoch(self):
        return len(self._dataloaders['train_loader'])

    def lr_finder(self, train_loader, start_lr, end_lr, exp_step_size, smoothing_window=50):
        self._dataloaders['train_loader'] = train_loader
        self.configure_optimizers()
        lr = start_lr
        pbar = tqdm(desc=f'max_lr: {end_lr}', dynamic_ncols=True)
        while lr <= end_lr:
            try:
                for batch in train_loader:
                    self._optimizers['optimizer'].param_groups[0]['lr'] = lr
                    pbar.set_postfix({'current lr:': self._optimizers['optimizer'].param_groups[0]['lr']})
                    loss = self.training_step(batch)
                    self._optimizers['optimizer'].zero_grad()
                    loss.backward()

                    self._optimizers['optimizer'].step()
                    self.log('loss', loss.cpu().item())
                    self.log('lr', lr)
                    if lr > end_lr:
                        break
                    lr *= 10 ** exp_step_size
                    pbar.update(1)
                self.on_epoch_end(self._log_dict)
            except Exception as e:
                print(e)
                break
        pbar.close()

        # plot lr vs loss
        import matplotlib.pyplot as plt

        # check directory for files and append a number if it exists
        file_num = 0
        if os.path.exists('lr_finder.png'):
            while os.path.exists(f'lr_finder{file_num}.png'):
                file_num += 1

        lr = np.array(self._log_dict['lr'])
        loss = np.array(self._log_dict['loss'])
        fig, ax = plt.subplots()
        ax.plot(lr, loss)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        pickle.dump(fig, open(f'lr_finder_{file_num}.pickle', 'wb'))
        fig.savefig(f'lr_finder_{file_num}.png')

        # compute savgol filter
        if smoothing_window % 2 == 0:
            smoothing_window += 1

        loss = savgol_filter(loss, smoothing_window, 3, mode='nearest')
        fig, ax = plt.subplots()
        ax.plot(lr, loss)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        pickle.dump(fig, open(f'lr_finder_smooth_{file_num}.pickle', 'wb'))
        fig.savefig(f'lr_finder_smooth_{file_num}.png')
        plt.show()


    def on_epoch_end(self, log_dict):
        pass
