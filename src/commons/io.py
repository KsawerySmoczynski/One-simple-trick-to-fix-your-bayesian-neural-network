import importlib

import torch as t
import yaml
from torch import nn


def parse_net_class(model_config_path: str):
    net_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)["model"]["model"]
    net_class, module = net_config["class_path"].split(".")[-1], ".".join(net_config["class_path"].split(".")[:-1])
    module = importlib.import_module(module)
    net_class = getattr(module, net_class)
    net = net_class(**net_config["init_args"])

    return net


def load_net(net: nn.Module, model_path: str, device: str, lightning_model: bool = True) -> nn.Module:
    if lightning_model:
        state_dict = {k.replace("model.", ""): v for k, v in t.load(model_path)["state_dict"].items()}
        net.load_state_dict(state_dict)
    else:
        net.load_state_dict(t.load(model_path))
    net = net.to(device)
    return net
