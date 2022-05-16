from typing import Tuple

import pyro
import torch
from pyro.infer import SVI, Predictive
from pyro.nn.module import PyroModule
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm

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
    log_file: str,
) -> Tuple[PyroModule, PyroModule]:
    with open(log_file, "w") as f:
        for e in range(epochs):
            loss = 0
            f.write(f"Epoch {e}\n")
            for X, y in tqdm(train_loader, desc=f"Epoch {e}", miniters=10):
                X = X.to(device)
                y = y.to(device)
                loss += svi.step(X, y)
            f.write(f"Loss: {loss / len(train_loader.dataset)}\n")
            if e % 5 == 0:
                predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("obs",))
                f.write(f"Start eval for epoch: {e}\n")
                evaluation(predictive, test_loader, metrics, device, f)
        return model, guide


def evaluation(predictive, dataloader, metrics, device, file):
    for X, y in dataloader:
        y = y.to(device)
        out = predictive(X.to(device))["obs"].T
        for metric in metrics:
            metric.update(out, y.to(device))
    # TODO report to tensorboard
    for metric in metrics:
        file.write(f"{metric.__class__.__name__} - {metric.compute():.4f}\n")
        metric.reset()
