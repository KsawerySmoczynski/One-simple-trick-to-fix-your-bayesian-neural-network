import importlib
from functools import reduce
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch import nn

from src.commons.utils import _rec_dict_merge


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


def initialize_object(object_dict: Dict):
    if "init_args" in object_dict:
        if isinstance(object_dict["init_args"], dict):
            for k, v in object_dict["init_args"].items():
                if isinstance(v, dict) and "class_path" in v:
                    object_dict["init_args"][k] = initialize_object(v)

    class_path = object_dict["class_path"]
    init_args = object_dict["init_args"] if "init_args" in object_dict else {}
    parts = class_path.split(".")
    module, net_class = ".".join(parts[:-1]), parts[-1]
    package = class_path.split(".")[0]
    module = importlib.import_module(module, package)
    cls = getattr(module, net_class)
    return cls(**init_args) if isinstance(init_args, dict) else cls(*init_args)


def parse_net_class(model_config_path: str):
    net_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)["model"]["model"]
    return initialize_object(net_config)


def load_net(net: nn.Module, model_path: str, device: str, lightning_model: bool = True) -> nn.Module:
    if lightning_model:
        state_dict = {k.replace("model.", ""): v for k, v in torch.load(model_path)["state_dict"].items()}
        net.load_state_dict(state_dict)
    else:
        net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    return net
