from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict

import pyro
import torch
from pyro.infer import SVI, Predictive
from torch.nn.utils import vector_to_parameters
from torch.utils.tensorboard import SummaryWriter

from src.commons.io import load_config, load_param_store, print_command, save_config
from src.commons.logging import save_metrics
from src.commons.pyro_training import evaluation, to_bayesian_model, train_loop
from src.commons.utils import (
    get_configs,
    get_metrics,
    seed_everything,
    traverse_config_and_initialize,
)


def main(config: Dict, args):
    model_config, data_config, metrics_config, training_config = get_configs(config)
    pyro.clear_param_store()
    seed_everything(training_config["seed"])
    training_config["num_samples"] = args.num_samples
    device = torch.device(training_config["device"])
    epochs = training_config["max_epochs"]
    point_estimate_model_config = {**model_config["model"]}
    model_config["model"] = traverse_config_and_initialize(model_config["model"])
    model = to_bayesian_model(**model_config, device=device)
    optimizer = traverse_config_and_initialize(model_config["optimizer"])
    criterion = traverse_config_and_initialize(model_config["criterion"])
    svi = SVI(model.model, model.guide, optimizer, loss=criterion)

    datamodule = traverse_config_and_initialize(data_config)
    train_loader = datamodule.train_dataloader()
    validation_loader = datamodule.validation_dataloader()
    test_loader = datamodule.test_dataloader()

    metrics = get_metrics(metrics_config, device)
    if args.monitor_metric and not args.monitor_metric in metrics:
        raise AttributeError(f"{args.monitor_metric} metric wasn't initialized check configs")

    dataset_name = str(datamodule)
    model_name = str(model.model)
    activation_name = str(model.model.model.activation)

    workdir = args.workdir / dataset_name / model_name / activation_name / datetime.now().strftime("%H:%M")
    workdir.mkdir(parents=True, exist_ok=True)
    save_config(config, workdir / "config.yaml")
    writer = SummaryWriter(workdir)

    metrics = get_metrics(metrics_config, device)

    model, guide = train_loop(
        model.model,
        model.guide,
        train_loader,
        validation_loader,
        svi,
        epochs,
        training_config["num_samples"],
        metrics,
        writer,
        workdir,
        device,
        args.evaluation_interval,
        args.monitor_metric,
        args.monitor_metric_mode,
        args.early_stopping_epochs,
    )

    print("Testing bayesian model...")
    bayesian_metrics_path = workdir / "bayesian_metrics.csv"
    for metric in metrics.values():
        metric.reset()
    if args.monitor_metric and not args.early_stopping_epochs:
        load_param_store(workdir)
    predictive = Predictive(model, guide=guide, num_samples=args.num_samples, return_sites=("_RETURN",))
    evaluation(predictive, test_loader, metrics, device)
    save_metrics(bayesian_metrics_path, metrics, dataset_name, model_name, activation_name, writer, stage="test")

    print("Testing point-estimate model...")
    for metric in metrics_config:
        metric["init_args"]["input_type"] = "none"
    point_estimate_metrics = get_metrics(metrics_config, device)

    net = traverse_config_and_initialize(point_estimate_model_config)
    net.to(device)
    vector_to_parameters(guide.loc, net.parameters())
    net.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            out = net(X)
            for metric in point_estimate_metrics.values():
                metric.update(out.exp(), y.to(device))
    print("Saving point-estimate model...")
    point_estimate_metrics_path = workdir / "point_estimate_metrics.csv"
    model_path = workdir / "point_estimate_params.pt"
    save_metrics(
        point_estimate_metrics_path,
        point_estimate_metrics,
        dataset_name,
        model_name,
        activation_name,
        writer,
        stage="point_estimate",
    )
    torch.save(net.state_dict(), model_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", nargs="*", type=str, help="Path to yaml with config")
    parser.add_argument(
        "--num-samples", type=int, default=100, help="How many samples should pyro use during prediction phase"
    )
    parser.add_argument("--monitor-metric", type=str, help="Save model depending on improvement on this metric")
    parser.add_argument("--monitor-metric-mode", type=str, help="Whether the metric is a stimulant or destimulant")
    parser.add_argument("--early-stopping-epochs", type=int, help="If to use early stopping during training phase")
    parser.add_argument(
        "--evaluation-interval", type=int, default=1, help="Every each epoch validation should be performed"
    )
    parser.add_argument("--workdir", type=Path, default=Path("logs"), help="Path to store training artifacts")
    args = parser.parse_args()
    metrics_valid = not (bool(args.monitor_metric) ^ bool(args.monitor_metric_mode))
    assert (
        metrics_valid
    ), "Arguments monitor metric and monitor metric mode should be either passed both or none of them should be passed"
    if args.early_stopping_epochs:
        assert args.early_stopping_epochs > 1, "Early stopping should be set up to > 1 epochs"
        assert (
            bool(args.monitor_metric) and bool(args.monitor_metric_mode) and args.early_stopping_epochs
        ), "Both metric to monitor and it's mode have to be set up while using early stopping"
    args.workdir = args.workdir / datetime.now().strftime("%Y%m%d")
    config = load_config(args.config)
    model_config, data_config, metrics_config, training_config = get_configs(config)
    print_command()

    # Will be moved
    training_config["num_samples"] = args.num_samples

    seed_everything(training_config["seed"])

    main(config, args)
