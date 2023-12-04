from training_module import TrainingModule
import torch.nn as nn
from scheduler import Scheduler
from resnet import ResNet
import torch
torch.manual_seed(0)


class NetworkTrainer(TrainingModule):
    def __init__(self, model, pre_training, device) -> None:
        super().__init__(model, device)
        self._model: ResNet = model
        self._pre_training = pre_training
        self._device = device
        self._gpu_float_2 = torch.tensor(2, dtype=torch.float, device=self._device)
        self._gpu_float_3 = torch.tensor(3, dtype=torch.float, device=self._device)

    def v_step(self, batch) -> float:
        state, value = batch
        state = state.to(self._device)
        value = value.to(self._device)
        value_hat = self._model(state)
        value_hat = value_hat.squeeze(1)
        value_loss = nn.functional.mse_loss(value_hat, value)
        del state, value, value_hat
        return value_loss

    def pv_step(self, batch) -> float:
        state, value, policy, legal_policy = batch
        state = state.to(self._device)
        policy = policy.to(self._device)
        legal_policy = legal_policy.to(self._device)
        value = value.to(self._device)
        policy_hat, value_hat = self._model(state)
        policy_hat = policy_hat.squeeze(1)
        value_hat = value_hat.squeeze(1)
        policy_hat = torch.mul(policy_hat, legal_policy)
        value_loss = nn.functional.mse_loss(value_hat, value)
        policy_loss = nn.functional.cross_entropy(policy_hat, policy)
        loss = torch.div(torch.add(policy_loss, value_loss), self._gpu_float_2)
        del state, policy, value, policy_hat, value_hat, policy_loss, value_loss, legal_policy
        return loss

    def training_step(self, batch):
        if self._pre_training:
            return self.v_step(batch)
        else:
            return self.pv_step(batch)

    def test_step(self, batch):
        if self._pre_training:
            return self.v_step(batch)
        else:
            return self.pv_step(batch)

    def configure_optimizers(self):
        # self.asgd()
        self.adamw()
        # self.rms()
        return self._step_list

    def asgd(self):
        optimizer = torch.optim.ASGD(
            self._model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )
        self._optimizers['optimizer'] = optimizer
        self._step_list.append(self._optimizers['optimizer'])
        # scheduler = Scheduler(
        #     optimizer=optimizer,
        #     num_steps=self.steps_per_epoch,
        #     max_lr=1e-4,
        #     min_lr=1e-6,
        #     pct_start=0.05,
        #     pct_max=0,
        #     annealing='cosine',
        #     conditional_wd=False
        # )
        # self._schedulers['scheduler'] = scheduler
        # self._step_list.append(scheduler)

    def adamw(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
            # betas=(0.85, 0.95),
            amsgrad=True,
        )
        self._optimizers['optimizer'] = optimizer
        self._step_list.append(self._optimizers['optimizer'])
        # scheduler = Scheduler(
        #     optimizer=optimizer,
        #     num_steps=self.steps_per_epoch,
        #     max_lr=1e-6,
        #     min_lr=1e-8,
        #     pct_start=0.05,
        #     pct_max=0.05,
        #     annealing='cosine',
        # )
        # self._schedulers['scheduler'] = scheduler
        # self._step_list.append(scheduler)

    def rms(self):
        optimizer = torch.optim.RMSprop(
            self._model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            momentum=0.9,
            centered=True,
        )
        self._optimizers['optimizer'] = optimizer
        self._step_list.append(self._optimizers['optimizer'])

    def on_epoch_end(self, log_dict):
        for dataloader in self._dataloaders.values():
            dataloader.restart()
        for scheduler in self._schedulers.values():
            scheduler.restart()
