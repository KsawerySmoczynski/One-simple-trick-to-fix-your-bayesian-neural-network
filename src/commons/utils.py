import random
from collections.abc import MutableMapping
from functools import reduce
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision import transforms

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


def _rec_dict_merge(d1: Dict, d2: Dict) -> Dict:
    for k, v in d1.items():
        if k in d2:
            if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                d2[k] = _rec_dict_merge(v, d2[k])
    d3 = d1.copy()
    d3.update(d2)
    return d3


def get_configs(config_paths: List[str]) -> Dict:
    if isinstance(config_paths, str):
        config_paths = [config_paths]
    configs = map(lambda path: yaml.safe_load(open(path, "r")), config_paths)
    config = reduce(_rec_dict_merge, configs)

    model = config["model"]
    data = config["data"]
    seed = config["seed_everything"]
    training = config["trainer"]
    training["seed"] = seed
    return model, data, training


def get_transforms(objective: str):
    # TODO extend to support 3channel transforms based on dataset name
    if objective == "classification":
        normalization = ((0.1307,), (0.3081,))
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*normalization),
                transforms.RandomAffine(degrees=(0, 70), translate=(0.1, 0.3), scale=(0.8, 1.2)),
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*normalization)])
    elif objective == "regression":
        print("Regression transforms not implemented, applying identity")
        # TODO add normalization transforms
        train_transform = lambda x: x
        test_transform = lambda x: x

    return train_transform, test_transform
