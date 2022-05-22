from typing import Tuple

import pyro
import torch
from pyro.infer import SVI, Predictive
from pyro.nn.module import PyroModule
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    writer: SummaryWriter,
    device: torch.DeviceObjType,
) -> Tuple[PyroModule, PyroModule]:
    for e in range(epochs):
        loss = 0
        for idx, (X, y) in tqdm(enumerate(train_loader), desc=f"Epoch {e}", miniters=10):
            X = X.to(device)
            y = y.to(device)
            step_loss = svi.step(X, y)
            loss += step_loss
            writer.add_scalar("train-loss/step", step_loss, (e + 1) * len(train_loader) + idx)
        writer.add_scalar("train-loss/epoch", loss, e + 1)
        if e % 5 == 0:
            predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("obs",))
            evaluation(e, predictive, test_loader, metrics, writer, device)
    return model, guide


def evaluation(epoch, predictive, dataloader, metrics, writer, device):
    for X, y in dataloader:
        y = y.to(device)
        out = predictive(X.to(device))["obs"].T
        for metric in metrics:
            metric.update(out, y.to(device))
    for metric in metrics:
        writer.add_scalar(f"eval/{metric.__class__.__name__}", metric.compute(), epoch)
        metric.reset()
