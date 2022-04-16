import pyro
import torch
from pyro.distributions import Categorical, Normal
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn.module import PyroModule, PyroSample, to_pyro_module_
from pyro.optim import MultiStepLR
from pyroapi import pyro
from torch import nn

# from pyro.optim import Adam
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms as T
from torchvision.datasets import MNIST

from src.models.classification import LeNet

DEVICE = torch.device("cuda:0")
train_dataset = MNIST("datasets", True, T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]), download=True)
test_dataset = MNIST("datasets", False, T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]), download=True)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=512, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2
)
test_loader = DataLoader(dataset=test_dataset, batch_size=6000, shuffle=False, num_workers=16)

inmemory_test_loader = [batch for batch in test_loader]

# Like this for cuda
# https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/vae.py


class CategoricalBayesianNet(PyroModule):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor, device: bool):
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std
        self.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        to_pyro_module_(model)

        for m in self.model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(
                    m, name, PyroSample(prior=Normal(self.mean, self.std).expand(value.shape).to_event(value.dim()))
                )

    def forward(self, x, y=None):
        pyro.module("model", self.model)
        logits = self.model(x)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", Categorical(logits=logits), obs=y)
        return logits


model = CategoricalBayesianNet(LeNet(1, 10, None), torch.tensor(0.0), torch.tensor(2.0), device=DEVICE)

# setup the optimizer
adam_params = {"lr": 1e-5, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)
# optimizer = MultiStepLR({'optimizer': Adam,
#                          'optim_args': {'lr': 1e-5, 'betas': (0.90, 0.999)},
#                          'gamma': 0.5, 'milestones': [15, 25]})
guide = AutoDiagonalNormal(model)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
epochs = 30

test_accuracy = Accuracy(num_classes=10)
num_test_samples = 100

pyro.clear_param_store()
for e in range(epochs):
    loss = 0
    for i, (X, y) in enumerate(train_loader):
        loss += svi.step(X.cuda(), y.cuda())
    print(loss)

    # TEST
    # predictive = Predictive(model, guide=guide, num_samples=num_test_samples, return_sites=("obs",))
    # for i, (X, y) in enumerate(inmemory_test_loader):
    #     output = torch.cat(
    #         [
    #             sample.unique(sorted=True, return_counts=True)[1] / num_test_samples
    #             for sample in predictive(X.cuda())["_RETURN"].T
    #         ]
    #     )
    #     output = output.reshape(6000, 10)
    #     acc = test_accuracy(output, y)
    #     print("siema")


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v.float(), 0).cpu().numpy().tolist(),
            "std": torch.std(v.float(), 0).cpu().numpy().tolist(),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0].cpu().numpy().tolist(),
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0].cpu().numpy().tolist(),
        }
    return site_stats


predictive = Predictive(model, guide=guide, num_samples=100, return_sites=("model.linear.6.weight", "obs", "_RETURN"))

samples = predictive(X[:1, ...].cuda())
output = model(X[:1, ...].cuda())
outputs = samples["obs"]
pred_summary = summary({k: v for k, v in samples.items() if k != "obs"})

import json

json.dump(pred_summary, open("summary.json", "w"), indent=4)
