from typing import Any

import torch as t
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.functional import accuracy


class BayesianModule(LightningModule):
    def __init__(self, model: nn.Module, lr: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # TODO add custom saving callback in CLI
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=True)
        self.log("train/accuracy", accuracy(y_hat, y), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("validation/loss", loss.item(), on_epoch=True)
        self.log("validation/accuracy", accuracy(y_hat, y), on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("test/loss", loss.item(), on_epoch=True)
        self.log("test/accuracy", accuracy(y_hat, y), on_epoch=True)

    def configure_optimizers(self):
        return t.optim.Adam(self.model.parameters(), lr=self.lr)
