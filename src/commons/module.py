from typing import Any

import pyro
import torch
from pyro.infer import SVI, Predictive
from pytorch_lightning import LightningModule
from torch import nn


class BayesianModule(LightningModule):
    def __init__(self, bayesian, model, optimizer, criterion, n_samples=100, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bayesian = bayesian
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_samples = n_samples
        self._configure_methods()
        #
        if bayesian:
            self.model.setup()  # pass device here should be available from trainer already.
        self.svi = SVI(self.model, self.model.guide, self.optimizer, loss=self.criterion) if bayesian else None

    def training_step(self, batch, batch_idx):
        return self.step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self.step("val", batch)

    def test_step(self, batch, batch_idx):
        return self.step("test", batch)

    def training_epoch_end(self, outputs):
        if self.bayesian:
            self.predictive = Predictive(self.model, guide=self.model.guide, num_samples=self.n_samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frwrd(x)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        self.bckwrd(loss, optimizer, optimizer_idx, *args, **kwargs)

    def configure_optimizers(self):
        if self.bayesian:
            pyro.clear_param_store()
        else:
            self.optimizer = self.optimizer(self.model.parameters())
        return self.optimizer

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs
    ):
        self.optim_step()

    def _step(self, stage: str, batch):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.criterion(y_hat, y)
        self.log(f"{stage}/loss", loss.item())
        return loss

    def _bayesian_step(self, stage: str, batch):
        X, y = batch
        loss = self._bayesian_train_step(self, X, y) if stage == "train" else self._bayesian_testval_step(X, y)
        return loss

    def _bayesian_train_step(self, stage, X, y):
        loss = self.svi.step(X, y)
        self.log(f"train/loss", loss.item())
        return loss.detach()

    def _bayesian_testval_step(self, X, y):
        out = self.predictive(X)
        # Report metrics
        # Calculate loss
        return None

    def _bayesian_forward(self, X):
        return self.predictive(X)

    def _bayesian_backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        pass

    def _bayesian_optim_step(self, *args, **kwargs):
        pass

    def _bayesian_optim_zero_grad(self, *args, **kwargs):
        pass

    def _configure_methods(self):
        self.step = self._bayesian_step if self.bayesian else self._step
        self.frwrd = self._bayesian_forward if self.bayesian else self.super.forward
        self.bckwrd = self._bayesian_backward if self.bayesian else self.super.backward
        self.optim_step = self._bayesian_optim_step if self.bayesian else self.super.optimizer_step
        self.optim_zero_grad = self._bayesian_optim_zero_grad if self.bayesian else self.super.optimizer_zero_grad
