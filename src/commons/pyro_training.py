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
    if monitor_metric:
        best_monitor_metric_value = get_monitored_metric_init_val(monitor_metric_mode)
    if early_stopping_epochs:
        no_improvement_epochs = 0
    for e in range(epochs):
        loss = training(svi, train_loader, e, writer, device)
        writer.add_scalar("train/loss-epoch", loss, e + 1)
        if (e + 1) % evaluation_interval == 0:
            predictive = Predictive(
                model,
                guide=guide,
                num_samples=num_samples,
                return_sites=(
                    "obs",
                    "_RETURN",
                ),
            )
            evaluation(predictive, valid_loader, metrics, device)
            if monitor_metric:
                current_monitor_metric_value = metrics[monitor_metric].compute().cpu()
                improved = monitor_metric_improvement(
                    best_monitor_metric_value, current_monitor_metric_value, monitor_metric_mode
                )
                if improved:
                    save_param_store(workdir)
                    best_monitor_metric_value = current_monitor_metric_value
                if early_stopping_epochs:
                    early_stop, no_improvement_epochs = eval_early_stopping(
                        early_stopping_epochs, no_improvement_epochs, improved
                    )
            report_metrics(metrics, "evaluation", e, writer)
            if monitor_metric and test_loader and save_predictions_config:
                if improved:
                    evaluation(predictive, test_loader, metrics, device, save_predictions_config)
                    report_metrics(metrics, "test-epoch", e, writer)
            if early_stopping_epochs:
                if early_stop:
                    print("STOPPING EARLY")
                    break

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
    save_predictions_config: Dict = None,
):
    if save_predictions_config:
        if Path(save_predictions_config["output_path"]).exists():
            with open(Path(save_predictions_config["output_path"]), "w") as f:
                f.write("")
    for idx, (X, y) in tqdm(enumerate(dataloader), desc=f"Evaluation", miniters=10):
        y = y.to(device)
        out = predictive(X.to(device))["_RETURN"]
        # change to "obs" if you want to obtain observations instead of probabilities
        if save_predictions_config:
            predictions = save_predictions_config["reduction"](out.cpu()).numpy()
            max_index = (idx + 1) * dataloader.batch_size
            max_index = len(dataloader.dataset) if max_index > len(dataloader.dataset) else max_index
            indices = np.arange(idx * dataloader.batch_size, max_index)
            with open(Path(save_predictions_config["output_path"]), "ab") as pred_file:
                np.savetxt(pred_file, np.c_[indices, predictions])
        for metric in metrics.values():
            metric.update(out, y)
