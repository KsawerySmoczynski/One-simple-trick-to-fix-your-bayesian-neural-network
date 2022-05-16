from functools import reduce
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch import nn

from src.commons.utils import _rec_dict_merge, initialize_object


def save_config(config: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as yaml_file:
        dump = yaml.dump(config, allow_unicode=True, encoding=None)
        yaml_file.write(dump)


def load_config(config_paths: List[str]):
    if isinstance(config_paths, str):
        config_paths = [config_paths]
    configs = map(lambda path: yaml.safe_load(open(path, "r")), config_paths)
    config = reduce(_rec_dict_merge, configs)
    return config


def parse_net_class(model_config_path: str, activation_path: str):
    activation_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)["activation"]
    net_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)["model"]["model"]
    net_config["init_args"]["activation"] = activation_config
    return initialize_object(net_config)


def load_net(net: nn.Module, model_path: str, device: str, lightning_model: bool = True) -> nn.Module:
    if lightning_model:
        state_dict = {k.replace("model.", ""): v for k, v in torch.load(model_path)["state_dict"].items()}
        net.load_state_dict(state_dict)
    else:
        net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    return net
