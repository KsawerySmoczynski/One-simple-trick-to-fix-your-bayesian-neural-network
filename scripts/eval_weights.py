from __future__ import print_function

import random
from argparse import ArgumentParser
from pathlib import Path

import json
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from src.commons.io import load_net, parse_net_class
from src.commons.plotting import plot_1d
from src.commons.utils import calculate_ll, modify_parameter, find_mass
from src.commons.utils import device

from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Predictive
from torch.nn.utils import vector_to_parameters
from torch.utils.tensorboard import SummaryWriter

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.commons.io import load_config, load_param_store, print_command, save_config
from src.commons.logging import save_metrics
from src.commons.pyro_training import evaluation, to_bayesian_model, train_loop
from src.commons.utils import (
    get_configs,
    get_metrics,
    seed_everything,
    traverse_config_and_initialize,
)


DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 42

t.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

parser = ArgumentParser()
parser.add_argument("--config", nargs="*", type=str, help="Path to yaml with config")
parser.add_argument("--save_dir", type=str, help="Path to directory where plots, etc. will be saved")
parser.add_argument("--net_path", type=str, help="Path to state dict")
parser.add_argument("--processes", type=int, default=2,  help="Number of processes for data loaders")
parser.add_argument("--override_plot_data", type=bool, default=False, help="Specifies if plotting data should be overridden")
parser.add_argument("--override_windows", type=bool, default=False, help="Specifies if likelihood windows shold be overridden")
parser.add_argument("--rate", type=int, default=200, help="Number of likelihood estimation points")
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

config = load_config(args.config)
model_config, data_config, metrics_config, training_config = get_configs(config)

net = traverse_config_and_initialize(model_config["model"])
datamodule = traverse_config_and_initialize(data_config)
# train_loader = datamodule.train_dataloader()
# validation_loader = datamodule.validation_dataloader()

# net = parse_net_class(args.net_config_path)
net.load_state_dict(torch.load(args.net_path))
net.to(device)
net.eval()

save_dir = f"{args.save_dir}/{net.__class__.__name__}/1d"

batch_size = args.batch_size
train_kwargs = {"batch_size": batch_size, 'shuffle': True}
if "cuda" in DEVICE:
    cuda_kwargs = {"num_workers": args.processes, "pin_memory": True, "prefetch_factor": 1}
    train_kwargs.update(cuda_kwargs)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST("datasets", train=True, transform=transform, download=True)
# train_limit = len(train_dataset)  # 1000
train_limit = 1000
train_dataset.data = train_dataset.data[:train_limit]
train_dataset.targets = train_dataset.targets[:train_limit]
train_loader = t.utils.data.DataLoader(train_dataset, **train_kwargs)

original_parameters = parameters_to_vector(net.parameters()).detach().clone()

Path(save_dir).mkdir(parents=True, exist_ok=True)
plt.hist(original_parameters.cpu().numpy(), bins=100)
plt.savefig(f"{save_dir}/parameters_hist.png")
plt.close()

# Draw random weights from each of the layers
draw_weight = lambda shape, range: t.tensor(np.array([np.random.choice(dim, range) for dim in shape])).T

# Num sampled weights
n_weights = 2
layers_shapes = {k: v.shape for k, v in net.state_dict().items() if "weight" in k}
sampled_indices = {k: draw_weight(v, n_weights) for k, v in layers_shapes.items()}

rate = args.rate
override_plot_data = args.override_plot_data
override_windows = args.override_windows

windows_path = f"{save_dir}/likelihood_mass.json"
windows = {}

plot_data_path = f"{save_dir}/plot_data.json" 
plot_data = {}

for layer_name, weights_indices in sampled_indices.items():
    windows[layer_name] = {}
    plot_data[layer_name] = {}
    print(layer_name)
    if layer_name == 'layer2.weight':
        for weight_idx in weights_indices:
            weight_name = str(weight_idx.tolist())
            original_weight = net.state_dict()[layer_name][tuple(weight_idx)].clone()
            
            df = None

            if exists(plot_data_path) and not override_plot_data:
                with open(plot_data_path) as f:
                    saved = json.load(f)
                    if layer_name in saved.keys() and weight_name in saved[layer_name].keys():
                        print(f"Restoring saved data for {layer_name} {weight_name}")
                        df = saved[layer_name][weight_name]

            if df is None:
                df = []
                # i = np.random.randint(0, original_parameters.numel()-1, size=1)
                # original_weight = original_parameters[i].item()
                # vector_to_parameters(original_parameters, net.parameters())

                left_window = None
                right_window = None

                print (exists(windows_path) and not override_windows)
                if exists(windows_path) and not override_windows:
                    print('lol')
                    with open(windows_path, 'r') as f:
                        saved = json.load(f)
                        if layer_name in saved.keys() and weight_name in saved[layer_name].keys():
                            print(f"Restoring saved windows for {layer_name} {weight_name}")
                            left_window, right_window = saved[layer_name][weight_name]

                if left_window is None:
                    # left_window, right_window = find_mass(net, layer_name, weight_idx, original_weight, train_loader)
                    left_window, right_window = 5, 1

                windows[layer_name][weight_name] = (left_window, right_window)

                for value in t.linspace(original_weight - left_window, original_weight + right_window, rate, device=DEVICE):
                    print(".", end="")
                    net.state_dict()[layer_name][tuple(weight_idx)] = value
                    # modify_parameter(net, i, value)
                    ll, good = calculate_ll(train_loader, net, DEVICE)
                    df.append((value.item() / 10, ll, good))
                net.state_dict()[layer_name][tuple(weight_idx)] = original_weight
                print("")

            plot_data[layer_name][weight_name] = df
            print(df)

            df = t.tensor(df).cpu().numpy()
            id = f"{layer_name}_{'_'.join(map(str, weight_idx.cpu().numpy()))}"
            # id = i
            plot_1d(df, original_weight.item(), id, train_limit, f"{save_dir}/{id}.png")

with open(windows_path, 'w') as f:
    json.dump(windows, f)

with open(plot_data_path, 'w') as f:
    json.dump(plot_data, f)