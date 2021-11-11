from typing import Any

import torch as t
from torch import nn
from pytorch_lightning import LightningModule

from models.FCN import FCN
from torchmetrics.functional import accuracy


class BayesianModule(LightningModule):
    def __init__(self, in_channels:int, n_classes:int, kernels_per_layer:int, lr:float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = FCN(in_channels, n_classes, kernels_per_layer)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("loss", loss.item(), on_step=True, on_epoch=True)
        self.log("accuracy", accuracy(y_hat, y), on_step=True, on_epoch=True)
        return loss
    
    def test_step(self):
        pass

    def configure_optimizers(self):
        return t.optim.Adam(self.model.parameters(), lr=self.lr)