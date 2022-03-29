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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def d(a):
  return torch.Tensor([a]).to(device)
