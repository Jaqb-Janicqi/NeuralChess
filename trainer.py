from resnet import ResNet
from db_dataloader import DataLoader
import torch
from scheduler import Scheduler
import torch.nn as nn
from training_module import TrainingModule


class NetworkTrainer(TrainingModule):
    def __init__(self, model, pre_training, device) -> None:
        super().__init__(model, device)
        self._model: ResNet = model
        self._pre_training = pre_training
        self._device = device

    def pre_training_step(self, batch) -> float:
        state, value = batch
        state = state.to(self._device)
        value = value.to(self._device)
        value_hat = self._model(state)
        value_loss = nn.functional.mse_loss(value_hat, value)
        self.log("value_loss", value_loss)
        return value_loss

    def true_training_step(self, batch) -> float:
        state, policy, value = batch
        state = state.to(self._device)
        policy = policy.to(self._device)
        value = value.to(self._device)
        policy_hat, value_hat = self._model(state)
        value_loss = nn.functional.mse_loss(value_hat, value)
        policy_loss = nn.functional.cross_entropy(policy_hat, policy)
        loss = policy_loss + value_loss
        self.log("policy_loss", policy_loss)
        self.log("value_loss", value_loss)
        self.log("loss", loss)
        return loss

    def training_step(self, batch):
        if self._pre_training:
            return self.pre_training_step(batch)
        else:
            return self.true_training_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=3e-3,
            weight_decay=0.3,
            betas=(0.85, 0.95),
        )
        self._optimizers['optimizer'] = optimizer
        scheduler = Scheduler(
            optimizer,
            num_steps=self.steps_per_epoch,
            max_lr=1e-3,
            min_lr=1e-5,
            pct_start=0.2,
            pct_max=0.1,
            annealing='cosine',
        )
        self._step_list.append(scheduler)

    def on_epoch_end(self, log_dict):
        for dataloader in self._dataloaders.values():
            dataloader.restart()