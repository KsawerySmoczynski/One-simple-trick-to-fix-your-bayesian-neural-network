from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy

from src.metrics.classification import ExpectedCalibrationError


class TrainingModule(LightningModule):
    def __init__(self, model: nn.Module, optimizer_args: dict, n_classes: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_args = optimizer_args
        self.optimizer = None
        self.accuracy = Accuracy(num_classes=n_classes)
        self.ece = ExpectedCalibrationError(num_classes=n_classes, input_type="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).exp()
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=True)
        self.log("train/accuracy", self.accuracy(y_hat, y), on_step=True, on_epoch=True)
        self.log("train/ECE", self.ece(y_hat, y), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).exp()
        loss = self.criterion(y_hat, y)
        self.log("validation/loss", loss.item(), on_epoch=True)
        self.log("validation/accuracy", self.accuracy(y_hat, y), on_epoch=True)
        self.log("validation/ECE", self.ece(y_hat, y), on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).exp()
        loss = self.criterion(y_hat, y)
        self.log("test/loss", loss.item(), on_epoch=True)
        self.log("test/accuracy", self.accuracy(y_hat, y), on_epoch=True)
        self.log("test/ECE", self.ece(y_hat, y), on_epoch=True)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_args)
        return self.optimizer
