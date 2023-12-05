from resnet import ResNet
from db_dataloader import DataLoader
import torch
import torch_directml as dml
from scheduler import Scheduler
import torch.nn as nn
import pytorch_lightning as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT


class LitResNet(pl.LightningModule):
    def __init__(self, resnet: ResNet, pre_training, device) -> None:
        super().__init__()
        self.__resnet = resnet
        self.__pre_training = pre_training
        self.__device = device

    def pre_training_step(self, batch) -> STEP_OUTPUT:
        id, fen, cp, prob, encoded = batch
        state = self.__resnet.to_tensor(encoded).to(self.__device)
        value = self.__resnet.to_tensor(prob).to(self.__device)
        policy_hat, value_hat = self.__resnet(state)
        value_loss = nn.functional.mse_loss(value_hat, value)
        self.log("value_loss", value_loss)
        return value_loss

    def true_training_step(self, batch) -> STEP_OUTPUT:
        state, policy, value = batch
        state = self.__resnet.to_tensor(state).to(self.__device)
        policy = self.__resnet.to_tensor(policy).to(self.__device)
        value = self.__resnet.to_tensor(value).to(self.__device)
        policy_hat, value_hat = self.__resnet(state)
        value_loss = nn.functional.mse_loss(value_hat, value)
        policy_loss = nn.functional.cross_entropy(policy_hat, policy)
        loss = policy_loss + value_loss
        self.log("policy_loss", policy_loss)
        self.log("value_loss", value_loss)
        self.log("loss", loss)
        return loss

    def training_step(self, batch) -> STEP_OUTPUT:
        if self.__pre_training:
            return self.pre_training_step(batch)
        else:
            return self.true_training_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.__resnet.parameters(),
            lr=1e-3,
            weight_decay=0.1,
            betas=(0.85, 0.95),
        )
        scheduler = Scheduler(
            optimizer,
            num_steps=self.trainer.estimated_stepping_batches,
            max_lr=1e-3,
            min_lr=1e-5,
            pct_start=0.2,
            pct_max=0.1,
            annealing='cosine',
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
