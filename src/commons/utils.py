import importlib
import random
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.models.normal import N


def modify_parameter(model, i, value):
    vec = parameters_to_vector(model.parameters())
    vec[i] = value
    vector_to_parameters(vec, model.parameters())


def calculate_ll(train_loader, model, device):
    test_loss = 0
    test_good = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += -F.nll_loss(F.log_softmax(output, dim=1), target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            test_good += pred.eq(target.view_as(pred)).sum().item()
    return test_loss, test_good


def fit_N(x, p):
    dist = N()
    optim = torch.optim.SGD(dist.parameters(), lr=0.4)
    for _ in range(1000):
        optim.zero_grad()
        xt = torch.from_numpy(x)
        out = dist(xt)
        loss = torch.mean((out - torch.from_numpy(p)) ** 2)
        loss.backward()
        optim.step()

    return out.cpu().detach().numpy()


def seed_everything(seed: int) -> int:
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if not (min_seed_value <= seed <= max_seed_value):
        print(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = np.randint(min_seed_value, max_seed_value)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


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


def _rec_dict_merge(d1: Dict, d2: Dict) -> Dict:
    for k, v in d1.items():
        if k in d2:
            if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                d2[k] = _rec_dict_merge(v, d2[k])
    d3 = d1.copy()
    d3.update(d2)
    return d3


def get_configs(config: Dict) -> Dict:
    # TODO change to proper handling with defaults and interpretable errors etc
    assert "model" in config
    assert "data" in config
    assert "seed_everything" in config
    assert "metrics" in config
    assert "trainer" in config

    model_config = config["model"]
    data_config = config["data"]
    seed = config["seed_everything"]
    metrics_config = config["metrics"]
    training_config = config["trainer"]
    training_config["seed"] = seed
    return model_config, data_config, metrics_config, training_config


def get_metrics(metrics_config: List) -> List:
    return [initialize_object(metric) for metric in metrics_config]


def traverse_config_and_initialize(iterable: Union[Dict, List, Tuple]):
    inpt = deepcopy(iterable)
    if isinstance(inpt, dict) and "class_path" in inpt:
        if "init_args" in inpt:
            inpt["init_args"] = traverse_config_and_initialize(inpt["init_args"])
        return initialize_object(inpt)
    elif isinstance(inpt, dict):
        for k, v in inpt.items():
            inpt[k] = traverse_config_and_initialize(v)
        return inpt
    elif isinstance(inpt, (list, tuple)):
        items = []
        for item in inpt:
            items.append(traverse_config_and_initialize(item))
        return items
    else:
        return inpt
