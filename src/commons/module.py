from functools import partial
from typing import Any, Union

import pyro
import torch
from pyro.infer import SVI, Predictive
from pytorch_lightning import LightningModule
from torch import nn

from models.bnn import bayesian_wrap
from src.commons.io import initialize_object


class BayesianModule(LightningModule):
    def __init__(
        self, bayesian: bool, model: nn.Module, optimizer, criterion, mean=0, std=1, n_samples=100, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bayesian = bayesian
        self.model = bayesian_wrap(model, mean, std) if bayesian else model
        self.optimizer = optimizer
        self.criterion = initialize_object(criterion["class_path"], criterion["init_args"])
        self.n_samples = n_samples
        self._configure_methods()
        self.svi = None

    def training_step(self, batch, batch_idx):
        return self.train_val_test_step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self.train_val_test_step("val", batch)

    def test_step(self, batch, batch_idx):
        return self.train_val_test_step("test", batch)

    def training_epoch_end(self, outputs):
        if self.bayesian:
            self.predictive = Predictive(self.model, guide=self.model.guide, num_samples=self.n_samples)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.frwrd(X)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        self.bckwrd(loss, optimizer, optimizer_idx, *args, **kwargs)

    def configure_optimizers(self):
        if self.bayesian:
            self.model.setup(self._device)  # pass device here should be available from trainer already.
            pyro.clear_param_store()
        else:
            self.optimizer["init_args"].update({"params": self.model.parameters()})
        self.optimizer = initialize_object(self.optimizer["class_path"], self.optimizer["init_args"])
        if self.bayesian:
            self.svi = SVI(self.model, self.model.guide, self.optimizer, loss=self.criterion)
            # return torch.optim.SGD([torch.tensor(1., requires_grad=True)], lr=0.1)
        return None

    # TODO and add zero grad
    def optimizer_step(self, *args, **kwargs):
        self.optim_step(*args, **kwargs)

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

    def _bayesian_optim_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, *args, **kwargs
    ):
        optimizer_closure()
        print("siema chyba nic nie będzie z tego Lightninga")
        pass

    def _bayesian_optim_zero_grad(self, *args, **kwargs):
        pass

    def _configure_methods(self):
        self.train_val_test_step = self._bayesian_step if self.bayesian else self._step
        self.frwrd = self._bayesian_forward if self.bayesian else self.model.forward
        self.bckwrd = self._bayesian_backward if self.bayesian else super().backward
        self.optim_step = self._bayesian_optim_step if self.bayesian else super().optimizer_step
        self.optim_zero_grad = self._bayesian_optim_zero_grad if self.bayesian else super().optimizer_zero_grad
