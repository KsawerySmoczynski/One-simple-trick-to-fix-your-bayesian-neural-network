from __future__ import print_function

import random
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from src.commons.io import load_net, parse_net_class
from src.commons.plotting import plot_1d, plot_2d
from src.commons.utils import calculate_ll, modify_parameter

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 42

t.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

parser = ArgumentParser()
parser.add_argument("net_path", type=str, help="Path to the pytorch lightning checkpoint or pytorch pickled state dict")
parser.add_argument("net_config_path", type=str, help="Path to config.yaml file from pytorch-lightning trainig")
parser.add_argument("processes", type=int, help="Number of processes for data loaders")
parser.add_argument("--save-dir", type=str, help="Path to directory where plots, etc. will be saved")
args = parser.parse_args()

net = parse_net_class(args.net_config_path)
net = load_net(net, args.net_path, device=DEVICE)
net.eval()

save_dir = f"{args.save_dir}/{net.__class__.__name__}/2d"

batch_size = 128
train_kwargs = {"batch_size": batch_size}
if "cuda" in DEVICE:
    cuda_kwargs = {"num_workers": args.processes, "pin_memory": True, "prefetch_factor": 1}
    train_kwargs.update(cuda_kwargs)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST("datasets", train=True, transform=transform)
train_limit = len(train_dataset)  # 1000
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
sampled_indices2 = {k: draw_weight(v, n_weights) for k, v in layers_shapes.items()}


window = 20
rate = 40

for (layer_name, weights_indices), weights_indices_2 in zip(sampled_indices.items(), sampled_indices2.values()):
    for weight_idx, weight_idx2 in zip(weights_indices, weights_indices_2):
        original_weight = net.state_dict()[layer_name][tuple(weight_idx)].clone()
        original_weight2 = net.state_dict()[layer_name][tuple(weight_idx2)].clone()
        df = []
        for value in t.linspace(original_weight - window, original_weight + window, rate, device=DEVICE):
            net.state_dict()[layer_name][tuple(weight_idx)] = value
            for value2 in t.linspace(original_weight2 - window, original_weight2 + window, rate, device=DEVICE):
                net.state_dict()[layer_name][tuple(weight_idx2)] = value2
                ll, good = calculate_ll(train_loader, net, DEVICE)
            net.state_dict()[layer_name][tuple(weight_idx2)] = original_weight2
            print(".", end="")
            df.append((value, value2, ll, good))
        net.state_dict()[layer_name][tuple(weight_idx)] = original_weight
        df = t.tensor(df).cpu().numpy()
        id1 = f"{layer_name}_{'_'.join(map(str, weight_idx.cpu().numpy()))}"
        id2 = f"{layer_name}_{'_'.join(map(str, weight_idx.cpu().numpy()))}"
        plot_2d(
            df, original_weight.item(), id1, original_weight2.item(), id2, train_limit, f"{save_dir}/{id1}x{id2}.png"
        )
