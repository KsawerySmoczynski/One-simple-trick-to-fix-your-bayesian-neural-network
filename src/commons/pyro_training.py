from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
from pyro.infer import SVI, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn.module import PyroModule
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric
from tqdm import tqdm

from src.commons.callbacks import EarlyStopping
from src.commons.io import save_param_store
from src.commons.logging import (
    get_monitored_metric_init_val,
    monitor_metric_improvement,
    report_metrics,
)
from src.commons.utils import eval_early_stopping
from src.models import CLASSIFICATION_MODELS, REGRESSION_MODELS
from src.models.bnn import BNNClassification, BNNContainer, BNNRegression


def to_bayesian_model(
    model: nn.Module, mean: float, std: float, device: torch.DeviceObjType, sigma_bound: float = 5.0, *args, **kwargs
) -> PyroModule:
    if model.__class__ in CLASSIFICATION_MODELS:
        model = BNNClassification(model, mean, std)
    elif model.__class__ in REGRESSION_MODELS:
        model = BNNRegression(model, mean, std, sigma_bound)
    else:
        raise NotImplementedError(f"Model {model.__class__.__name__} is currently unsupported in bayesian setting")
    model.setup(device)
    guide = AutoDiagonalNormal(model)

    return BNNContainer(model, guide)


def train_loop(
    model,
    guide,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    svi: SVI,
    epochs: int,
    num_samples: int,
    metrics: Tuple[Metric],
    writer: SummaryWriter,
    workdir: Path,
    device: torch.DeviceObjType,
    evaluation_interval: int = 1,
    monitor_metric: str = None,
    monitor_metric_mode: str = None,
    early_stopping_epochs: int = False,
    test_loader: DataLoader = None,
    save_predictions_config: DataLoader = None,
) -> Tuple[PyroModule, PyroModule]:
    early_stopping = None
    if monitor_metric:
        early_stopping = EarlyStopping(monitor_metric, monitor_metric_mode, early_stopping_epochs, path=workdir)
    for e in range(epochs):
        loss = training(svi, train_loader, e, writer, device)
        writer.add_scalar("train/loss-epoch", loss, e + 1)
        if (e + 1) % evaluation_interval == 0:
            predictive = Predictive(
                model,
                guide=guide,
                num_samples=num_samples,
                return_sites=("_RETURN",),
            )
            evaluation(predictive, valid_loader, metrics, device)
            if monitor_metric:
                if early_stopping.improved(metrics):
                    with open(workdir / "best_epoch.txt", "w") as f:
                        f.write(str(e))
            report_metrics(metrics, "evaluation", e, writer, reset=True)
            # if monitor_metric and test_loader and save_predictions_config:
            #     if improved:
            evaluation(predictive, test_loader, metrics, device)
            report_metrics(metrics, "test", e, writer, reset=True)
            if monitor_metric:
                if early_stopping.early_stop:
                    print("STOPPING EARLY")
                    import sys

                    sys.exit(0)
    import sys

    sys.exit(0)
    return model, guide


def training(svi: SVI, train_loader: Iterator, epoch: int, writer: SummaryWriter, device: torch.DeviceObjType):
    loss = 0
    for idx, (X, y) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc=f"Training epoch {epoch}", miniters=10
    ):
        X = X.to(device)
        y = y.to(device)
        step_loss = svi.step(X, y)
        loss += step_loss
        batch_index = (epoch + 1) * len(train_loader) + idx
        writer.add_scalar("train/loss-step", step_loss, batch_index)
    return loss


def evaluation(
    predictive: Predictive,
    dataloader: Iterator,
    metrics: Dict,
    device: torch.DeviceObjType,
):
    for idx, (X, y) in tqdm(enumerate(dataloader), desc=f"Evaluation", miniters=10):
        y = y.to(device)
        out = predictive(X.to(device))["_RETURN"]
        for metric in metrics.values():
            metric.update(out, y)
