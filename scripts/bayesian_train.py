from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pyro
import torch
from pyro.infer import SVI

from src.commons.data import get_dataloaders, get_datasets, get_transforms
from src.commons.io import load_config, save_config
from src.commons.pyro_training import to_bayesian_model, train
from src.commons.utils import (
    get_configs,
    get_metrics,
    seed_everything,
    traverse_config_and_initialize,
)


def main(model_config, data_config, metrics_config, training_config):
    pyro.clear_param_store()
    device = torch.device("cuda") if (training_config["gpus"] != 0) else torch.device("cpu")
    epochs = training_config["max_epochs"]
    model_config["model"] = traverse_config_and_initialize(model_config["model"])
    model = to_bayesian_model(**model_config)
    model.setup(device)
    optimizer = traverse_config_and_initialize(model_config["optimizer"])
    criterion = traverse_config_and_initialize(model_config["criterion"])

    data_config["train_transform"], data_config["test_transform"] = get_transforms(data_config)

    data_config["train_dataset"], data_config["test_dataset"] = get_datasets(**data_config)
    train_loader, test_loader = get_dataloaders(**data_config)

    svi = SVI(model, model.guide, optimizer, loss=criterion)

    eval_metrics = get_metrics(metrics_config)

    train(
        model, model.guide, train_loader, test_loader, svi, epochs, training_config["num_samples"], eval_metrics, device
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", nargs="*", type=str, help="Path to yaml with config")
    parser.add_argument(
        "--num-samples", type=int, default=100, help="How many samples should pyro use during prediciton phase"
    )  # TO be moved to configs
    parser.add_argument("--workdir", type=Path, default=Path("logs"), help="Path to store training artifacts")

    args = parser.parse_args()
    config = load_config(args.config)
    model_config, data_config, metrics_config, training_config = get_configs(
        config
    )  # TODO Add overriding feature of config entries with cmdline arguments
    save_config(config, args.workdir / datetime.now().strftime("%Y%m%d-%H:%M") / "config.yaml")

    # Will be moved
    training_config["num_samples"] = args.num_samples

    seed_everything(training_config["seed"])

    # DEF LOGGER
    main(model_config, data_config, metrics_config, training_config)
