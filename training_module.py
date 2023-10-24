import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml as dml
from tqdm import tqdm


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
                map(lambda obj: obj.step(), self._step_list)

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

    def lr_finder(self, train_loader, start_lr, end_lr, exp_step_size, ma_window=100):
        self._dataloaders['train_loader'] = train_loader
        self.configure_optimizers()
        lr = start_lr
        pbar = tqdm(desc=f'max_lr: {end_lr}', dynamic_ncols=True)
        while lr <= end_lr:
            for batch in train_loader:
                self._optimizers['optimizer'].param_groups[0]['lr'] = lr
                pbar.set_postfix({'current lr:': lr})
                loss = self.training_step(batch)
                self._optimizers['optimizer'].zero_grad()
                loss.backward()

                map(lambda obj: obj.step(), self._step_list)
                self.log('loss', loss.cpu().item())
                self.log('lr', lr)
                if lr > end_lr:
                    break
                lr *= 10 ** exp_step_size
                pbar.update(1)
            self.on_epoch_end(self._log_dict)
        pbar.close()

        # plot lr vs loss
        import matplotlib.pyplot as plt
        plt.plot(self._log_dict['lr'], self._log_dict['loss'])
        plt.savefig('lr_finder.png')
        plt.close()

        # compute moving average of loss
        avg_loss = []
        for i in range(len(self._log_dict['loss']) - ma_window):
            avg_loss.append(
                sum(self._log_dict['loss'][i:i + ma_window]) / ma_window)
        # make lr and loss same length
        self._log_dict['lr'] = self._log_dict['lr'][ma_window:]

        # plot lr vs avg_loss
        plt.plot(self._log_dict['lr'], avg_loss)
        plt.savefig('lr_finder_avg.png')
        plt.close()

    def on_epoch_end(self, log_dict):
        pass
