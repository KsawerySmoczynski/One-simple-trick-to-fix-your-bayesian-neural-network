#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pyro
import torch
import torch.nn as nn
import torch.optim as optim
from pyro.distributions import Categorical, Normal
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import SGD, Adam
from torchvision import datasets, transforms

from scripts.pyro.model import (
    BayesianFCNNet,
    FCNNet,
    guide_definition,
    model_definition,
)
from scripts.pyro.utils import calculate_test_acc

num_iterations = 101
num_samples = 10
hidden_size = 32
batch_size = 512
train_limit = 6000

# activation = 'relu'
activation = "leaky"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST("datasets", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("datasets", train=False, transform=transform)
train_dataset.data = train_dataset.data[:train_limit]
train_dataset.targets = train_dataset.targets[:train_limit]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8192, shuffle=True)
inmemory_iter = [a for a in enumerate(train_loader, 1)]
inmemory_test_loader = [a for a in enumerate(test_loader)]

print("######")
print(activation)
print("######")

model = BayesianFCNNet(28 * 28, hidden_size, 10, activation=activation)
model.cuda()
log_softmax = nn.LogSoftmax(dim=1).cuda()
softplus = torch.nn.Softplus().cuda()

optim = Adam({"lr": 0.001})
# optim = SGD({"lr": 0.00005})

# model = pyro.nn.module.PyroModule[FCNNet](28 * 28, hidden_size, 10, activation=activation)
# model = model_definition(net)
# guide = guide_definition(net)
guide = AutoDiagonalNormal(model)
guide.cuda()

svi = SVI(model, guide, optim, loss=Trace_ELBO())

predictive = Predictive(model, guide=guide, num_samples=num_samples)

pyro.clear_param_store()
calculate_test_acc(inmemory_test_loader, predictive)
for j in range(1, num_iterations + 1):
    loss = 0
    for batch_id, data in inmemory_iter:
        # calculate the loss and take a gradient step
        loss += svi.step(data[0].view(-1, 28 * 28).cuda(), data[1].cuda())
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train
    print("Epoch ", j, " Loss ", total_epoch_loss_train)
    if j % 1 == 0:
        calculate_test_acc(inmemory_test_loader, predictive)
calculate_test_acc(inmemory_test_loader, predictive)
