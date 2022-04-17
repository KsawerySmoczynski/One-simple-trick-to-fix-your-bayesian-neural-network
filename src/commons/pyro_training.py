from typing import Tuple

import numpy as np
import pyro
import torch

torch.set_printoptions(precision=10)
from pyro.infer import SVI, Predictive
from pyro.nn.module import PyroModule
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric

from src.models import CLASSIFICATION_MODELS, REGRESSION_MODELS
from src.models.bnn import BNNClassification, BNNRegression


def to_bayesian_model(
    model: nn.Module, mean: float, std: float, sigma_bound: float = 5.0, *args, **kwargs
) -> PyroModule:
    if model.__class__ in CLASSIFICATION_MODELS:
        return BNNClassification(model, mean, std)
    elif model.__class__ in REGRESSION_MODELS:
        return BNNRegression(model, mean, std, sigma_bound)
    else:
        raise NotImplementedError(f"Model {model.__class__.__name__} is currently unsupported in bayesian setting")


def get_objective(model):
    if isinstance(model, BNNClassification):
        return "classification"
    elif isinstance(model, BNNRegression):
        return "regression"
    else:
        raise Exception("Unknown objective")


def train(
    model,
    guide,
    train_loader: DataLoader,
    test_loader: DataLoader,
    svi: SVI,
    epochs: int,
    num_samples: int,
    metrics: Tuple[Metric],
    device: torch.DeviceObjType,
) -> Tuple[PyroModule, PyroModule]:
    pyro.clear_param_store()
    for e in range(epochs):
        print("Epoch: ", e)
        loss = 0
        for i, (X, y) in enumerate(train_loader):
            if i % 25 == 0:
                print(f"step {i}/{len(train_loader)}")
            X = X.to(device)
            y = y.to(device)
            loss += svi.step(X, y)
        print("Loss:", loss / len(train_loader.dataset))
        predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("obs",))
        print(f"Start eval for epoch: {e}")
        evaluation(predictive, test_loader, metrics, device)
    return model, guide


def evaluation(predictive, dataloader, metrics, device):
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        out = predictive(X)["obs"].T
        for metric in metrics:
            metric.update(out, y)
    # TODO report to tensorboard
    for metric in metrics:
        print(f"{metric.__class__.__name__} - {metric.compute():.4f}")
        metric.reset()
