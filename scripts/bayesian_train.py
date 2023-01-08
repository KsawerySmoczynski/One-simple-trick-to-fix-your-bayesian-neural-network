from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import pyro
import torch
from torch import nn
import math
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
    device = torch.device("cuda") if (training_config["gpus"] != 0) else torch.device("cpu")
    epochs = training_config["max_epochs"]
    point_estimate_model_config = {**model_config["model"]}
    net = traverse_config_and_initialize(model_config["model"])
    model_config["model"] = traverse_config_and_initialize(model_config["model"])
    model = to_bayesian_model(**model_config, device=device)
    optimizer = traverse_config_and_initialize(model_config["optimizer"])
    criterion = traverse_config_and_initialize(model_config["criterion"])
    svi = SVI(model.model, model.guide, optimizer, loss=criterion)

    datamodule = traverse_config_and_initialize(data_config)
    train_loader = datamodule.train_dataloader()
    validation_loader = datamodule.validation_dataloader()
    test_loader = datamodule.test_dataloader()

    print(len(train_loader), len(validation_loader), len(test_loader))

    metrics = get_metrics(metrics_config, device)
    if args.monitor_metric and not args.monitor_metric in metrics:
        raise AttributeError(f"{args.monitor_metric} metric wasn't initialized check configs")

    dataset_name = str(datamodule)
    model_name = str(model.model)
    activation_name = str(model.model.model.activation)
    seed_name = str(training_config["seed"])

    workdir = args.workdir / dataset_name / model_name / activation_name / seed_name / datetime.now().strftime("%H:%M")
    workdir.mkdir(parents=True, exist_ok=True)
    save_config(config, workdir / "config.yaml")
    writer = SummaryWriter(workdir)

    metrics = get_metrics(metrics_config, device)

    # TODO -> get rid of
    # to_hist, test_loader and save_predictions_config from train_loop
    def to_hist(x, bins):
        bins = torch.arange(bins)
        match = x[:, :, None] == bins[None, :]
        return match.sum(1) / x.shape[1]  # normalized

    save_predictions_config = {"reduction": partial(to_hist, bins=10), "output_path": f"{workdir}/results.csv"}

    print(args.deterministic_training)
    print(args.task)
    
    if args.task == 'regression' and args.deterministic_training:
        deterministic_path = workdir / "deterministic_results.txt"
        
        criterion = nn.MSELoss() 
        optimizer = torch.optim.Adam(net.parameters()) 
        net.cuda()
        
        print("DETERMINISTIC TRAINING")
        for epoch in range(50):
            with open(deterministic_path, 'a') as f:
                net.eval()
                eval_losses = []
                for inputs, labels in validation_loader:
                    inputs = inputs.float().cuda()
                    labels = labels.float().cuda()
                    outputs = net(inputs)

                    eval_losses.append(((outputs - labels)**2).mean())

                f.write(f"Validation RMSE: {math.sqrt(sum(eval_losses) / len(validation_loader))}\n")

                print(f"EPOCH: {epoch + 1}")
  
                net.train()
                f.write(f"Epoch: {epoch + 1}\n")
                train_losses = []
                for inputs, labels in train_loader:
                    inputs = inputs.float().cuda()
                    labels = labels.float().cuda()
                    optimizer.zero_grad()
                 
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                 
                    train_losses.append(loss.item())

                f.write(f"Tain RMSE: {math.sqrt(sum(train_losses) / len(train_loader))}\n")

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
        test_loader,
        save_predictions_config,
    )

    print("Testing bayesian model...")
    bayesian_metrics_path = workdir / "bayesian_metrics.csv"
    for metric in metrics.values():
        metric.reset()
    if args.monitor_metric and not args.early_stopping_epochs:
        load_param_store(workdir)
    predictive = Predictive(model, guide=guide, num_samples=args.num_samples, return_sites=("obs",))
    evaluation(predictive, test_loader, metrics, device)
    save_metrics(bayesian_metrics_path, metrics, dataset_name, model_name, activation_name, writer, stage="test")

    if args.test_point_estimate:
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
                if args.task == 'classification':
                    out = out.exp()
                for metric in point_estimate_metrics.values():
                    metric.update(out, y.to(device))
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
    # parser.add_argument("--task-type", type=str, help="Regression or Classification")
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
    parser.add_argument("--test-point-estimate", action="store_true")
    parser.add_argument("--deterministic-training", action="store_true")
    parser.add_argument("--seed", type=int, help="seed for randomization")
    parser.add_argument("--lr", type=float, help="learning rate for training")
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
    # args.workdir = args.workdir / datetime.now().strftime("%Y%m%d")
    config = load_config(args.config)

    if 'regression' in config['model']['model']['class_path']:
        args.task = 'regression'
    elif 'classification' in config['model']['model']['class_path']:
        args.task = 'classification'

    if args.seed is not None:
        config["seed_everything"] = args.seed

    if args.lr is not None:
        config["model"]["optimizer"]["init_args"][0]["lr"] = args.lr

    print_command()

    main(config, args)
