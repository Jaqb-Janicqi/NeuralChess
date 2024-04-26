import pytorch_lightning as L
from torch import optim, nn


class LitResNet(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = nn.functional.mse_loss(y_hat, y)
        super().log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = nn.functional.mse_loss(y_hat, y)
        super().log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6, amsgrad=True)
        reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-8)
        return {'optimizer': optimizer, 'lr_scheduler': reduce_lr, 'monitor': 'val_loss'}
