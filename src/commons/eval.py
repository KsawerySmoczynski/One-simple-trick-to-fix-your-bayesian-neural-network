import numpy as np
import torch as t
from torch.nn.utils import vector_to_parameters

from src.commons.plotting import plot_1d, plot_2d
from src.commons.utils import calculate_ll, modify_parameter


def eval_1d(original_parameters, model, inds, device, full_train_loader, train_limit, save_path=None):
    window = 20
    rate = 50

    for idx, i in enumerate(inds, 1):
        val = original_parameters[i].cpu()
        print(i, val)
        vector_to_parameters(original_parameters, model.parameters())
        df = []
        for value in t.linspace(val - window, val + window, rate, device=device):
            print(".", end="")
            modify_parameter(model, i, value)
            ll, good = calculate_ll(full_train_loader, model)
            df.append((value, ll, good))
        df = t.tensor(df).cpu().numpy()
        plot_1d(df, val, i, train_limit, save_path)


def eval_2d(original_parameters, model, inds, device, full_train_loader, train_limit, save_path=None):
    window = 10
    rate = 20
    model.to(device)
    for idx, (i1, i2) in enumerate(inds, 1):
        val1 = original_parameters[i1]
        val2 = original_parameters[i2]
        print(i1, val1)
        print(i2, val2)
        vector_to_parameters(original_parameters, model.parameters())
        df = []
        for value1 in t.linspace(val1 - window, val1 + window, rate, device=device):
            modify_parameter(model, i1, value1)
            for value2 in t.linspace(val2 - window, val2 + window, rate, device=device):
                modify_parameter(model, i2, value2)
                ll, good = calculate_ll(full_train_loader, model)
                df.append((value1, value2, ll, good))
            print(".", end="")
        df = t.tensor(df).cpu().numpy()
        plot_2d(df, val1.cpu(), i1, val2.cpu(), i2, train_limit)
