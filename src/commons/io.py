import importlib
from typing import Dict

import torch as t
import yaml
from torch import nn


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
        state_dict = {k.replace("model.", ""): v for k, v in t.load(model_path)["state_dict"].items()}
        net.load_state_dict(state_dict)
    else:
        net.load_state_dict(t.load(model_path))
    net = net.to(device)
    return net
