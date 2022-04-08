import importlib
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import torch as t
import yaml
from torch import nn


def initialize_object(class_path: str, init_args: dict):
    parts = class_path.split(".")
    package = None
    module, cls = ".".join(parts[:-1]), parts[-1]
    if parts[0] != "src":
        package = class_path.split(".")[0]
        # module = ".".join(module.split(".")[1:])
    module = importlib.import_module(module, package)
    cls = getattr(module, cls)
    if isinstance(init_args, (List, Tuple)):
        return cls(*init_args)
    elif isinstance(init_args, Dict):
        return cls(**init_args)


def parse_net_class(model_config_path: str):
    net_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)["model"]["model"]
    return initialize_object(net_config["class_path"], net_config["init_args"])


def load_net(net: nn.Module, model_path: str, device: str, lightning_model: bool = True) -> nn.Module:
    if lightning_model:
        state_dict = {k.replace("model.", ""): v for k, v in t.load(model_path)["state_dict"].items()}
        net.load_state_dict(state_dict)
    else:
        net.load_state_dict(t.load(model_path))
    net = net.to(device)
    return net
