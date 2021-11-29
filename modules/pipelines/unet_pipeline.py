from torch import optim
import pytorch_lightning as pl
from modules.losses.iou_loss import IoULoss


class UnetPipeline(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.iou_loss = IoULoss()
        self.config = config


    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.config["learning_rate"])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_hat = self.forward(train_batch['image'])
        loss = self.iou_loss(x_hat, train_batch['heatmaps'])
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x_hat = self.forward(val_batch['image'])
        loss = self.iou_loss(x_hat, val_batch['heatmaps'])
        self.log('val_loss', loss,  on_step=False, on_epoch=True, prog_bar=True, logger=True)
