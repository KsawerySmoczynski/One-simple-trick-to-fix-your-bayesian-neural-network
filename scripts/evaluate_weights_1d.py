from __future__ import print_function

import json
import os
import random
from argparse import ArgumentParser
from os.path import exists
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm

from src.commons.io import load_net, parse_net_class
from src.commons.plotting import plot_1d
from src.commons.utils import calculate_ll, find_mass, modify_parameter

device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

parser = ArgumentParser()
# parser.add_argument("save_dir", type=str, help="Path to directory where plots, etc. will be saved")
parser.add_argument("net_path", type=str, help="Path to the pytorch lightning checkpoint or pytorch pickled state dict")
parser.add_argument("net_config_path", type=str, help="Path to config.yaml file from pytorch-lightning trainig")
# parser.add_argument("activation_path", type=str, help="Path to config.yaml file with activation function")
parser.add_argument("--processes", type=int, default=2, help="Number of processes for data loaders")
parser.add_argument(
    "--override_plot_data", type=bool, default=False, help="Specifies if plotting data should be overridden"
)
parser.add_argument(
    "--override_windows", type=bool, default=False, help="Specifies if likelihood windows shold be overridden"
)
parser.add_argument("--rate", type=int, default=25, help="Number of likelihood estimation points")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--in-memory", action="store_true", help="Whether to load dataset in memory")
args = parser.parse_args()

net = parse_net_class(args.net_config_path)
net = load_net(net, args.net_path, device=device, lightning_model=("ckpt" in args.net_config_path))
net.eval()

save_dir = Path(args.net_path).parent
save_dir.mkdir(parents=True, exist_ok=True)

batch_size = args.batch_size
train_kwargs = {"batch_size": batch_size, "shuffle": True}
if "cuda" in device:
    cuda_kwargs = {"num_workers": args.processes, "pin_memory": True, "prefetch_factor": 1}
    train_kwargs.update(cuda_kwargs)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = None
if "FashionMNIST" in args.net_path:
    train_dataset = datasets.FashionMNIST("datasets", train=True, transform=transform, download=True)
elif "MNIST" in args.net_path:
    train_dataset = datasets.MNIST("datasets", train=True, transform=transform, download=True)

train_limit = 6000
train_dataset.data = train_dataset.data[:train_limit]
train_dataset.targets = train_dataset.targets[:train_limit]
train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
if args.in_memory:
    train_loader = [b for b in train_loader]

original_parameters = parameters_to_vector(net.parameters()).detach().clone()

plt.hist(original_parameters.cpu().numpy(), bins=100)
plt.savefig(f"{save_dir}/parameters_hist.png")
plt.close()

# Draw random weights from each of the layers
draw_weight = lambda shape, range: torch.tensor(np.array([np.random.choice(dim, range) for dim in shape])).T

# Num sampled weights
n_weights = 2
layers_shapes = {k: v.shape for k, v in net.state_dict().items() if "weight" in k}
sampled_indices = {k: draw_weight(v, n_weights) for k, v in layers_shapes.items()}

rate = args.rate
override_plot_data = args.override_plot_data
override_windows = args.override_windows

windows_path = save_dir / "likelihood_mass.json"
windows = {}
plot_data_path = save_dir / "plot_data.json"
plot_data = {}
for layer_name, weights_indices in sampled_indices.items():
    windows[layer_name] = {}
    plot_data[layer_name] = {}
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

            if exists(windows_path) and not override_windows:
                with open(windows_path, "r") as f:
                    saved = json.load(f)
                    if layer_name in saved.keys() and weight_name in saved[layer_name].keys():
                        print(f"Restoring saved windows for {layer_name} {weight_name}")
                        left_window, right_window = saved[layer_name][weight_name]

            if left_window is None:
                left_window, right_window = find_mass(
                    net, layer_name, weight_idx, original_weight, train_loader, device
                )

            windows[layer_name][weight_name] = (left_window, right_window)

            for value in tqdm(
                torch.linspace(original_weight - left_window, original_weight + right_window, rate, device=device),
                desc=f"Weight {weight_idx+1}/{len(weights_indices)}",
            ):
                net.state_dict()[layer_name][tuple(weight_idx)] = value
                # modify_parameter(net, i, value)
                ll, good = calculate_ll(train_loader, net, device)
                df.append((value.item(), ll, good))
            net.state_dict()[layer_name][tuple(weight_idx)] = original_weight

        plot_data[layer_name][weight_name] = df

        df = torch.tensor(df).cpu().numpy()
        id = f"{layer_name}_{'_'.join(map(str, weight_idx.cpu().numpy()))}"
        plot_1d(df, original_weight.item(), id, train_limit, f"{save_dir}/{id}.png")

with open(windows_path, "w") as f:
    json.dump(windows, f)

with open(save_dir / "plot_data.json", "w") as f:
    json.dump(plot_data, f)
