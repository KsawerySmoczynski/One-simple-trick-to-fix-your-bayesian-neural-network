from pathlib import Path
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter


def report_metrics(metrics: Dict, stage: str, epoch: int, writer: SummaryWriter, reset: bool = True) -> None:
    for metric_name, metric in metrics.items():
        writer.add_scalar(f"{stage}/{metric_name}", metric.compute(), epoch)
        if reset:
            metric.reset()


def get_monitored_metric_init_val(monitor_metric_mode: str) -> torch.Tensor:
    monitor_metric_value = (
        torch.tensor(torch.finfo(float).min) if monitor_metric_mode == "max" else torch.tensor(torch.finfo(float).max)
    )
    return monitor_metric_value


def monitor_metric_improvement(
    monitor_metric_value: float, current_metric_value: float, monitor_metric_mode: str
) -> torch.Tensor:
    if monitor_metric_mode == "max":
        improved = monitor_metric_value < current_metric_value
    elif monitor_metric_mode == "min":
        improved = monitor_metric_value > current_metric_value

    return improved


def save_metrics(
    metrics_path: Path,
    metrics: Dict,
    dataset_name: str,
    model_name: str,
    activation_name: str,
    writer: SummaryWriter = None,
    stage: str = None,
) -> None:
    with open(metrics_path, "w") as f:
        f.write("dataset,model,activation,metric,metric_value\n")
        for metric_name, metric in metrics.items():
            metric_readout = f"{dataset_name},{model_name},{activation_name},{metric_name},{metric.compute():.4f}\n"
            if writer and stage:
                writer.add_scalar(f"{stage}/{metric_name}", metric.compute(), 0)
            f.write(metric_readout)
