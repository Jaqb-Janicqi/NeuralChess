import chess
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools as it
import seaborn as sns
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm
import torch_directml as dml
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
import numpy as np
import torch
import optuna

from mcts_node import Node
torch.manual_seed(0)


class TrainingModule:
    def __init__(self, model, device=dml.device()):
        self._model: nn.Module = model
        self._device = device
        self._optimizers = {}  # {key: optimizer}
        self._schedulers = {}  # {key: scheduler}
        self._log_dict = {}  # {key: [value]}
        self._step_list = []  # [class]
        self._dataloaders = {}  # {key: dataloader}
        self._save_path = ''

    def log(self, key, value):
        if key not in self._log_dict:
            self._log_dict[key] = []
        self._log_dict[key].append(value)

    def training_step(self, batch):
        x, y = batch
        x = x.to(self._device)
        y = y.to(self._device)
        y_hat = self._model(x)
        loss = F.cross_entropy(y_hat, y)
        del x, y, y_hat
        return loss

    def test_step(self, batch):
        x, y = batch
        x = x.to(self._device)
        y = y.to(self._device)
        y_hat = self._model(x)
        loss = F.cross_entropy(y_hat, y)
        del x, y, y_hat
        return loss

    def convert_batch(self, batch):
        fens, labels = batch
        encoded = []
        for fen in fens[0]:
            state = chess.Board(fen)
            node = Node(1, state, {})
            encoded.append(node.encoded)
        new_labels = []
        encoded = np.array(encoded)
        labels = np.array(labels[0])
        encoded = torch.tensor(encoded).float().to(self._device)
        labels = torch.tensor(labels).float().to(self._device)
        batch = (encoded, labels)
        return batch

    def fit(self, train_loader, val_loader=None, max_epochs=None, early_stopping=0,
            save_attrs={}, path=None, resume_model_path=None, log_plot=False,
            resume=True, resume_optimizer=True, reduece_on_plateau=True):
        self._dataloaders['train_loader'] = train_loader
        if val_loader is not None:
            self._dataloaders['val_loader'] = val_loader
        if len(self._optimizers) == 0:
            self.configure_optimizers()
        start_epoch = 0
        if resume:
            if resume_model_path is not None:
                start_epoch = self.load(resume_model_path, resume_optimizer)
            else:
                start_epoch = self.load_best(path, resume_optimizer)
        if max_epochs is None:
            max_epochs = sys.maxsize ** 10

        recent_loss = 0
        for epoch in range(start_epoch+1, max_epochs+1):
            pbar = tqdm(desc=f'Epoch {epoch}', leave=False,
                        dynamic_ncols=True, total=len(train_loader))

            self._model.train()
            for batch in train_loader:
                # batch = self.convert_batch(batch)
                loss = self.training_step(batch)
                self.log('train_loss', loss.item())
                self._optimizers['optimizer'].zero_grad()
                loss.backward()
                del loss

                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
                for func in self._step_list:
                    func.step()

                pbar.update(1)
                lr = self._optimizers['optimizer'].param_groups[0]['lr']
                pbar.set_description(
                    f'Epoch {epoch}, lr: {lr}')

                if len(self._log_dict['train_loss']) > 100:
                    recent_loss = sum(
                        self._log_dict['train_loss'][-100:]) / 100
                else:
                    recent_loss = sum(
                        self._log_dict['train_loss']) / len(self._log_dict['train_loss'])
                pbar.set_postfix(
                    {'recent_averaged_loss': recent_loss})

            if val_loader is None:
                self.log('avg_loss',
                         sum(self._log_dict['train_loss']) / len(self._log_dict['train_loss']))
            else:
                self._model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        # batch = self.convert_batch(batch)
                        val_loss = self.test_step(batch).item()
                        self.log('val_loss', val_loss)
                self.log('avg_loss',
                         sum(self._log_dict['val_loss']) / len(self._log_dict['val_loss']))

            if early_stopping > 0:
                if len(self._log_dict['avg_loss']) > early_stopping:
                    if self._log_dict['avg_loss'][-1] > np.all(self._log_dict['avg_loss'][-(early_stopping+1):-2]):
                        break

            if reduece_on_plateau:
                if len(self._log_dict['avg_loss']) > 1:
                    if self._log_dict['avg_loss'][-1] > self._log_dict['avg_loss'][-2]:
                        self.reduce_scheduler_lr()

            if recent_loss == 0:
                recent_loss = sum(self._log_dict['train_loss']) / \
                    len(self._log_dict['train_loss'])
            if path is not None:
                self.save(path, epoch, recent_loss, save_attrs)
            self.on_epoch_end(self._log_dict)
            pbar.close()

            # save training loss
            smoothing_window = 51
            loss = savgol_filter(
                self._log_dict['train_loss'], smoothing_window, 3, mode='nearest')
            fig, ax = plt.subplots()
            ax.plot(loss)
            ax.set_xlabel('Batch')
            # ax.set_ylabel('Loss')
            if log_plot:
                ax.set_yscale('log')
            ax.set_title('Training Loss')
            plt.savefig(f'{path}training_loss.pdf')
            plt.close()

            # load best model
            if path is not None:
                self.load_best(path, resume_optimizer)

    def save(self, path, epoch, loss, model_attrs={}):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizers['optimizer'].state_dict(),
            'log_dict': self._log_dict,
        }
        for key in model_attrs.keys():
            save_dict[key] = model_attrs[key]
        torch.save(save_dict, f'{path}model_{epoch}_{loss}.pth')

    def load(self, path, resume_optimizer):
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        if resume_optimizer:
            self._optimizers['optimizer'].load_state_dict(
                checkpoint['optimizer_state_dict'])
        self._log_dict = checkpoint['log_dict']
        epoch = checkpoint['epoch']
        return epoch

    def load_best(self, path, resume_optimizer):
        best_loss = sys.maxsize
        best_path = ''
        for file in os.listdir(path):
            if file.endswith('.pth'):
                split_filename = file.split('_')
                loss = float(split_filename[-1].strip('.pth'))
                if loss < best_loss:
                    best_loss = loss
                    best_path = f'{path}{file}'
        return self.load(best_path, resume_optimizer)

    def configure_optimizers(self):
        self._step_list = []
        self._optimizers['optimizer'] = torch.optim.AdamW(
            self._model.parameters(),
        )
        self._step_list.append(self._optimizers['optimizer'])

    @property
    def steps_per_epoch(self):
        return len(self._dataloaders['train_loader'])

    def on_epoch_end(self, log_dict):
        pass

    def reduce_scheduler_lr(self, factor=0.1):
        for scheduler in self._schedulers.values():
            max_lr = scheduler.max_lr
            min_lr = scheduler.min_lr
            scheduler.max_lr = max_lr * factor
            scheduler.min_lr = min_lr * factor
        for optimizer in self._optimizers.values():
            for param_group in optimizer.param_groups:
                param_group['lr'] *= factor

    def lr_finder(self, train_loader, start_lr, end_lr, exp_step_size, smoothing_window=51):
        self._dataloaders['train_loader'] = train_loader
        self.configure_optimizers()
        lr = start_lr
        pbar = tqdm(desc=f'max_lr: {end_lr}', dynamic_ncols=True)
        while lr <= end_lr:
            try:
                # sweep lr each batch
                for batch in train_loader:
                    self._optimizers['optimizer'].param_groups[0]['lr'] = lr
                    pbar.set_postfix(
                        {'current lr:': self._optimizers['optimizer'].param_groups[0]['lr']})
                    loss = self.training_step(batch)
                    self._optimizers['optimizer'].zero_grad()
                    loss.backward()
                    self._optimizers['optimizer'].step()

                    self.log('loss', loss.item())
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

        # check directory for files and append a number if it exists
        file_num = 0
        if os.path.exists(f'{self._save_path}lr_finder.png'):
            while os.path.exists(f'{self._save_path}lr_finder{file_num}.png'):
                file_num += 1

        lr = np.array(self._log_dict['lr'])
        loss = np.array(self._log_dict['loss'])

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
        fig.savefig(f'{self._save_path}lr_finder_{file_num}.pdf')

    def grid_search(self, dataloader, val_dataloader, params, smoothing_window=50):
        self._dataloaders['train_loader'] = dataloader
        self._dataloaders['val_loader'] = val_dataloader
        param_keys = list(params.keys())
        if smoothing_window % 2 != 0:
            smoothing_window += 1
        self.configure_optimizers()
        for obj in self._step_list:
            if not isinstance(obj, torch.optim.Optimizer):
                self._step_list.remove(obj)

        # save empty model
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizers['optimizer'].state_dict(),
        }, f'grid_search.pth')

        num_steps = 1
        for key in param_keys:
            num_steps *= len(params[key])
        pbar = tqdm(
            desc=f'Grid Search', dynamic_ncols=True, total=num_steps)

        meshgrid = list(it.product(*params.values()))

        data_dict = {}
        for key in param_keys:
            data_dict[key] = []
        data_dict['val_loss'] = []
        data_dict['train_loss'] = []

        for i in range(num_steps):
            current_params = {}
            for j, key in enumerate(param_keys):
                current_params[key] = meshgrid[i][j]
                self._optimizers['optimizer'].param_groups[0][key] = meshgrid[i][j]
                data_dict[key].append(meshgrid[i][j])
            pbar.set_postfix(current_params)

            self.fit(dataloader, val_dataloader, max_epochs=1, plot=False)
            data_dict['val_loss'].append(self._log_dict['val_loss'])
            data_dict['train_loss'].append(self._log_dict['train_loss'])
            self._log_dict['val_loss'] = []
            self._log_dict['train_loss'] = []
            pbar.update(1)

            # reset model and optimizer
            self._model.load_state_dict(
                torch.load('grid_search.pth')['model_state_dict'])
            self._optimizers['optimizer'].load_state_dict(
                torch.load('grid_search.pth')['optimizer_state_dict'])
        pbar.close()

        # check directory for files and append a number if it exists
        file_num = 0
        while os.path.exists(f'grid_search{file_num}.csv'):
            file_num += 1

        dframe = pd.DataFrame(data_dict)
        dframe.to_csv(f'grid_search{file_num}.csv', index=False)

        sns.set_theme()
        dframe['val_loss'] = dframe['val_loss'].str.strip('[]').astype(float)
        dframe.sort_values(by=['betas'], inplace=True)
        color_norm = plt.Normalize(
            vmin=dframe['val_loss'].min(), vmax=dframe['val_loss'].max())

        categorical_count = len(dframe['betas'].unique())
        nrows = np.ceil(categorical_count/3).astype(int)
        fig, axes = plt.subplots(nrows, 3, sharex=False,
                                 sharey=False, squeeze=False)
        fig.set_figwidth(3*8)
        fig.set_figheight(nrows*8)
        axes = axes.flatten()
        for i, beta in enumerate(dframe['betas'].unique()):
            df = dframe[dframe['betas'] == beta]
            sns.heatmap(df.pivot_table(index='wd', columns='lr', values='val_loss'), cmap="viridis", norm=color_norm, cbar_kws={
                        'label': 'Validation Loss'}, ax=axes[i], annot=True, fmt='.6f', annot_kws={'fontsize': 8})
            axes[i].set_title(f'Beta: {beta}')
        # remove empty plots
        for i in range(categorical_count, len(axes)):
            fig.delaxes(axes[i])
        plt.savefig(f'grid_search{file_num}.pdf')

        for i, beta in enumerate(dframe['betas'].unique()):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # set size
            fig.set_figwidth(20)
            fig.set_figheight(10)
            df = dframe[dframe['betas'] == beta]
            num_lines = len(df)
            colors = sns.color_palette("viridis", num_lines)
            ax.set_prop_cycle(color=colors)
            for i in range(len(df)):
                t_loss = df.iloc[i]['train_loss']
                t_loss = t_loss.strip('[]').replace(' ', '').split(',')
                t_loss = [float(i) for i in t_loss]
                t_loss = savgol_filter(t_loss, 51, 3, mode='nearest')
                ax.plot(
                    t_loss, label=f'lr: {df.iloc[i]["lr"]}, wd: {df.iloc[i]["wd"]}')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(f'Beta: {beta}')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.savefig(f'beta_{beta}.pdf')
            plt.close()

    def _optuna_objective(self, trial):
        # reset model and optimizer
        self._model.load_state_dict(
            torch.load(f'{self._save_path}grid_search.pth')['model_state_dict'])
        self._optimizers['optimizer'].load_state_dict(
            torch.load(f'{self._save_path}grid_search.pth')['optimizer_state_dict'])
        running_loss = []

        # generate parameters
        params = {
            'lr': trial.suggest_float('lr', 1e-8, 1, log=True),
            'lambd': trial.suggest_float('lambd', 1e-8, 1, log=True),
            'alpha': trial.suggest_float('alpha', 0, 1, log=False),
            'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1, log=True),
            # 'betas': trial.suggest_categorical('betas', [str((0.85, 0.95)), str((0.9, 0.99))])
        }
        # tmp = params['betas'].strip('()').split(',')
        # params['betas'] = (float(tmp[0]), float(tmp[1]))
        for key in params.keys():
            self._optimizers['optimizer'].param_groups[0][key] = params[key]

        total_steps = len(self._dataloaders['train_loader'])

        train_bar = tqdm(
            desc=f'Optuna Trial',
            dynamic_ncols=True,
            total=total_steps,
            leave=False
        )

        self._model.train()
        for step, batch in enumerate(self._dataloaders['train_loader']):
            loss = self.training_step(batch)
            self._optimizers['optimizer'].zero_grad()
            loss.backward()
            loss = loss.item()
            running_loss.append(loss)

            train_bar.update(1)
            for func in self._step_list:
                func.step()

            # average over 100k samples
            if len(running_loss) > 1/(len(batch[0])//16)*6400:
                running_loss.pop(0)
            avg_loss = sum(running_loss) / len(running_loss)
            train_bar.set_postfix({'loss': loss, 'average loss': avg_loss})
            trial.report(avg_loss, step)

            if trial.should_prune():
                train_bar.close()
                self.on_epoch_end(self._log_dict)
                raise optuna.exceptions.TrialPruned()

        train_bar.close()
        self.on_epoch_end(self._log_dict)

        # save model
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizers['optimizer'].state_dict(),
        }, f'{self._save_path}grid_search{trial.number}.pth')
        return avg_loss

    def optuna_study(self, train_loader, study_name, n_trials=100, timeout=None):
        self._dataloaders['train_loader'] = train_loader
        self.configure_optimizers()
        for obj in self._step_list:
            if not isinstance(obj, torch.optim.Optimizer):
                self._step_list.remove(obj)
        self._save_path = f'{study_name}/'
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)

        # save empty model
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizers['optimizer'].state_dict(),
        }, f'{self._save_path}grid_search.pth')

        study = optuna.create_study(
            study_name=study_name, load_if_exists=True, direction='minimize', storage=f'sqlite:///{self._save_path}{study_name}.db')
        study.optimize(self._optuna_objective,
                       n_trials=n_trials, timeout=timeout)

        if os.path.exists(f'{study_name}.csv'):
            dframe = pd.read_csv(f'{study_name}.csv')
            dframe = dframe.append(
                pd.DataFrame(study.trials_dataframe()), ignore_index=True)
        else:
            dframe = pd.DataFrame(study.trials_dataframe())
        dframe.to_csv(f'{self._save_path}{study_name}.csv', index=False)


if __name__ == "__main__":
    dframe = pd.read_csv('grid_search4.csv')
    for i, beta in enumerate(dframe['betas'].unique()):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # set size
        fig.set_figwidth(20)
        fig.set_figheight(10)
        df = dframe[dframe['betas'] == beta]
        df.sort_values(by=['val_loss'], inplace=True)
        num_lines = len(df)
        colors = sns.color_palette("viridis", num_lines)
        ax.set_prop_cycle(color=colors)
        for i in range(len(df)):
            t_loss = df.iloc[i]['train_loss']
            t_loss = t_loss.strip('[]').replace(' ', '').split(',')
            t_loss = [float(i) for i in t_loss]
            t_loss = savgol_filter(t_loss, 51, 3, mode='nearest')
            if np.any(t_loss > 0.6):
                continue
            ax.plot(
                t_loss, label=f'lr: {df.iloc[i]["lr"]}, wd: {df.iloc[i]["wd"]}')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'Beta: {beta}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(f'beta_{beta}.pdf')
        plt.close()
