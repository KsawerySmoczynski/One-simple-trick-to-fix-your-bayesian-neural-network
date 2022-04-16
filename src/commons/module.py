from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.functional import accuracy

from src.commons.io import initialize_object


class TrainingModule(LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: dict, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        y_hat = self(x)
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
        self.optimizer["init_args"]["params"] = self.model.parameters()
        self.optimizer = initialize_object(self.optimizer)
        return self.optimizer
