from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
from pyro.infer import SVI

from src.commons.data import get_dataloaders, get_datasets
from src.commons.io import initialize_object, load_config, save_config
from src.commons.pyro_training import get_objective, to_bayesian_model, train
from src.commons.utils import get_configs, get_transforms, seed_everything


def main(model_config, data_config, metrics_config, training_config):
    device = torch.device("cuda") if (training_config["gpus"] != 0) else torch.device("cpu")
    epochs = training_config["max_epochs"]
    model_config["model"] = initialize_object(model_config["model"])
    model = to_bayesian_model(**model_config)
    model.setup(device)
    optimizer = initialize_object(model_config["optimizer"])
    criterion = initialize_object(model_config["criterion"])

    objective = get_objective(model)
    data_config["train_transform"], data_config["test_transform"] = get_transforms(objective)

    train_dataset, test_dataset = get_datasets(**data_config)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, **data_config)

    svi = SVI(model, model.guide, optimizer, loss=criterion)

    eval_metrics = [initialize_object(metric) for metric in metrics_config]
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
