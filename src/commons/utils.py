import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.models.normal import N

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def d(a):
    return torch.Tensor([a]).to(device)

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

def find_mass(net, layer, idx, val, train_loader):
    thres = 0.01
    mult = 1.1
    init_window = 0.1
    max_window = 10000

    logp ,_ = calculate_ll(train_loader, net, device)

    right_window = init_window
    while right_window < max_window:
        print(',', end='')
        new_val = val + right_window
        net.state_dict()[layer][tuple(idx)] = new_val 
        ll, _ = calculate_ll(train_loader, net, device)

        if np.exp(ll - logp) < thres:
            break

        else:
            right_window *= 1.1

    print("")

    left_window = init_window
    while left_window < max_window:
        print(',', end='')
        new_val = val - left_window
        net.state_dict()[layer][tuple(idx)] = new_val 
        ll, _ = calculate_ll(train_loader, net, device)

        if np.exp(ll - logp) < thres:
            break

        else:
            left_window *= 1.1

    print("")
    print("calculated likelihood mass for layer:")
    print(layer)

    return left_window, right_window

def fit_N(x, p):
    dist = N()
    optim = torch.optim.Adam(dist.parameters())
    for _ in range(1000):
        optim.zero_grad()
        xt = torch.from_numpy(x)
        out = dist(xt)
        loss = torch.mean((out - torch.from_numpy(p)) ** 2)
        loss.backward()
        optim.step()

    # print(dist.mu)
    return out.cpu().detach().numpy()

def fit_sigma(x, p):
    mu_idx = np.argmax(p)
    mu = x[mu_idx]

    min_err = None
    max_sigma = 100000
    sigma = 0.1

    while sigma < max_sigma:
        pn = torch.exp(-((torch.Tensor(x) - mu) ** 2 / (2 * sigma ** 2)))
        # normalize
        pn = pn / pn[mu_idx]
        err = torch.mean((pn - torch.from_numpy(p)) ** 2)

        if min_err is not None and err > min_err:
            break
        
        if min_err is None or err < min_err:
            min_err = err

        sigma *= 1.1

    return pn


