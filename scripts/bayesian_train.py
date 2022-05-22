from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict

import pyro
import torch
from pyro.infer import SVI
from torch.nn.utils import vector_to_parameters
from torch.utils.tensorboard import SummaryWriter

from src.commons.data import get_dataloaders, get_datasets, get_transforms
from src.commons.io import load_config, save_config
from src.commons.pyro_training import to_bayesian_model, train
from src.commons.utils import (
    get_configs,
    get_metrics,
    seed_everything,
    traverse_config_and_initialize,
)


def main(config: Dict, workdir: Path):
    model_config, data_config, metrics_config, training_config = get_configs(config)
    pyro.clear_param_store()
    seed_everything(training_config["seed"])
    training_config["num_samples"] = args.num_samples

    device = torch.device("cuda") if (training_config["gpus"] != 0) else torch.device("cpu")
    epochs = training_config["max_epochs"]
    point_estimate_model_config = {**model_config["model"]}
    model_config["model"] = traverse_config_and_initialize(model_config["model"])
    model = to_bayesian_model(**model_config)
    model.setup(device)
    optimizer = traverse_config_and_initialize(model_config["optimizer"])
    criterion = traverse_config_and_initialize(model_config["criterion"])
    svi = SVI(model, model.guide, optimizer, loss=criterion)

    data_config["train_transform"], data_config["test_transform"] = get_transforms(data_config)
    data_config["train_dataset"], data_config["test_dataset"] = get_datasets(**data_config)
    train_loader, test_loader = get_dataloaders(**data_config)

    eval_metrics = get_metrics(metrics_config)
    for metric in eval_metrics:
        metric.set_device(device)

    workdir = (
        workdir
        / train_loader.dataset.__class__.__name__
        / model.model.__class__.__name__
        / model.model.activation.__class__.__name__
    )
    workdir.mkdir(parents=True, exist_ok=True)
    save_config(config, workdir / "config.yaml")

    writer = SummaryWriter(workdir)

    model, guide = train(
        model,
        model.guide,
        train_loader,
        test_loader,
        svi,
        epochs,
        training_config["num_samples"],
        eval_metrics,
        writer,
        device,
    )

    print("Evaluating point-estimate model...")
    for metric in metrics_config:
        metric["init_args"]["input_type"] = "none"
    test_metrics = get_metrics(metrics_config)
    for metric in test_metrics:
        metric.set_device(device)

    net = traverse_config_and_initialize(point_estimate_model_config)
    net.to(device)
    vector_to_parameters(guide.loc, net.parameters())
    net.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            out = net(X)
            for metric in test_metrics:
                metric.update(out.exp(), y.to(device))

    print("Saving model...")
    metrics_path = workdir / "metrics.txt"
    model_path = workdir / "params.pt"

    with open(metrics_path, "w") as f:
        for metric in test_metrics:
            metric_readout = f"{metric.__class__.__name__} - {metric.compute():.4f}\n"
            writer.add_scalar(f"point_estimate/{metric.__class__.__name__}", metric.compute(), 0)
            f.write(metric_readout)
    torch.save({"state_dict": net.state_dict()}, model_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", nargs="*", type=str, help="Path to yaml with config")
    parser.add_argument(
        "--num-samples", type=int, default=100, help="How many samples should pyro use during prediciton phase"
    )
    parser.add_argument("--workdir", type=Path, default=Path("logs"), help="Path to store training artifacts")

    args = parser.parse_args()
    args.workdir = args.workdir / datetime.now().strftime("%Y%m%d")
    config = load_config(args.config)

    main(config, args.workdir)
