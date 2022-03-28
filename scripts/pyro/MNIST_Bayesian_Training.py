#!/usr/bin/env python
# coding: utf-8


###### pyro should be in version 0.2.1 ########

import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from pyro.distributions import Categorical, Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

###### pyro should be in version 0.2.1 ########


class FCNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, x):
        x = torch.flatten(x, 1)
        output = self.fc1(x)
        if self.activation == "relu":
            output = F.relu(output)
        elif self.activation == "leaky":
            output = F.leaky_relu(output, negative_slope=0.5)
        else:
            raise NotImplementedError("Unknown activation:", self.activation)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output


def model(x_data, y_data):
    scale_mult = 100.0
    fc1w_prior = Normal(
        loc=torch.zeros_like(net.fc1.weight).cuda(), scale=scale_mult * torch.ones_like(net.fc1.weight).cuda()
    )
    fc1b_prior = Normal(
        loc=torch.zeros_like(net.fc1.bias).cuda(), scale=scale_mult * torch.ones_like(net.fc1.bias).cuda()
    )

    outw_prior = Normal(
        loc=torch.zeros_like(net.out.weight).cuda(), scale=scale_mult * torch.ones_like(net.out.weight).cuda()
    )
    outb_prior = Normal(
        loc=torch.zeros_like(net.out.bias).cuda(), scale=scale_mult * torch.ones_like(net.out.bias).cuda()
    )

    priors = {"fc1.weight": fc1w_prior, "fc1.bias": fc1b_prior, "out.weight": outw_prior, "out.bias": outb_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    lhat = log_softmax(lifted_reg_model(x_data))

    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)


def guide(x_data, y_data):

    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight).cuda()
    fc1w_sigma = torch.randn_like(net.fc1.weight).cuda()
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias).cuda()
    fc1b_sigma = torch.randn_like(net.fc1.bias).cuda()
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.out.weight).cuda()
    outw_sigma = torch.randn_like(net.out.weight).cuda()
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.out.bias).cuda()
    outb_sigma = torch.randn_like(net.out.bias).cuda()
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {"fc1.weight": fc1w_prior, "fc1.bias": fc1b_prior, "out.weight": outw_prior, "out.bias": outb_prior}

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()


def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return torch.argmax(mean, dim=1)


def calculate_test_acc():
    correct = 0
    total = 0
    for j, data in inmemory_test_loader:
        images, labels = data
        predicted = predict(images.view(-1, 28 * 28).cuda())
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
    print("accuracy: %d %%" % (100 * correct / total))


num_iterations = 101
num_samples = 10
hidden_size = 42
batch_size = 512
train_limit = 6000

# activation = 'relu'
activation = "leaky"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("../data", train=False, transform=transform)
train_dataset.data = train_dataset.train_data[:train_limit]
train_dataset.targets = train_dataset.train_labels[:train_limit]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8192, shuffle=True)
inmemory_iter = [a for a in enumerate(train_loader, 1)]
inmemory_test_loader = [a for a in enumerate(test_loader)]

# for _ in range(5):
#     for activation in ['relu', 'leaky']:
#     # for activation in ['leaky', 'relu']:
print("######")
print(activation)
print("######")
net = FCNNet(28 * 28, hidden_size, 10, activation=activation)
net.cuda()
log_softmax = nn.LogSoftmax(dim=1).cuda()
softplus = torch.nn.Softplus().cuda()

# optim = Adam({"lr": 0.001})
optim = SGD({"lr": 0.00005})

svi = SVI(model, guide, optim, loss=Trace_ELBO())
calculate_test_acc()
for j in range(1, num_iterations + 1):
    loss = 0
    for batch_id, data in inmemory_iter:
        # calculate the loss and take a gradient step
        loss += svi.step(data[0].view(-1, 28 * 28).cuda(), data[1].cuda())
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train
    print("Epoch ", j, " Loss ", total_epoch_loss_train)
    if j % 1 == 0:
        calculate_test_acc()
calculate_test_acc()


# train_limit 600
# r: accuracy: 77 %
# l: accuracy: 79 %


# ######
# leaky
# ######
# accuracy: 9 %
# Epoch  1  Loss  7781.167767016236
# accuracy: 88 %
# Epoch  2  Loss  7536.092252284758
# accuracy: 90 %
# Epoch  3  Loss  7492.473983689086
# accuracy: 91 %
# Epoch  4  Loss  7472.4932125878495
# accuracy: 91 %
# Epoch  5  Loss  7450.396791742754
# accuracy: 91 %
# Epoch  6  Loss  7433.21803365206
# accuracy: 92 %
# Epoch  7  Loss  7423.984343831134
# accuracy: 92 %
# Epoch  8  Loss  7412.521513465866
# accuracy: 92 %
# Epoch  9  Loss  7399.077827001738
# accuracy: 92 %
# Epoch  10  Loss  7388.452180169885
# accuracy: 92 %
# Epoch  11  Loss  7383.5393468670845
# accuracy: 92 %
# Epoch  12  Loss  7367.353721424278
# accuracy: 92 %
# Epoch  13  Loss  7367.780766725215
# accuracy: 92 %
# Epoch  14  Loss  7352.903488879625
# accuracy: 92 %
# Epoch  15  Loss  7344.9674231809295
# accuracy: 92 %
# Epoch  16  Loss  7340.157877805623
# accuracy: 92 %
# Epoch  17  Loss  7332.449975951751
# accuracy: 92 %
# Epoch  18  Loss  7321.8990278904275
# accuracy: 92 %
# Epoch  19  Loss  7321.809581075191
# accuracy: 92 %
# Epoch  20  Loss  7311.237704139233
# accuracy: 92 %
# Epoch  21  Loss  7303.856645090119
# accuracy: 92 %
# Epoch  22  Loss  7295.705256912406
# accuracy: 93 %
# Epoch  23  Loss  7289.659198091658
# accuracy: 92 %
# Epoch  24  Loss  7284.563497493155
# accuracy: 93 %
# Epoch  25  Loss  7279.538372230721
# accuracy: 93 %
# Epoch  26  Loss  7272.080593576233
# accuracy: 93 %
# Epoch  27  Loss  7267.945162123632
# accuracy: 92 %
# Epoch  28  Loss  7261.418912718852
# accuracy: 92 %
# Epoch  29  Loss  7254.923328325033
# accuracy: 92 %
# Epoch  30  Loss  7246.238408061688
# accuracy: 92 %
# Epoch  31  Loss  7240.580533446948
# accuracy: 93 %
# accuracy: 93 %

# ######
# relu
# ######
# accuracy: 8 %
# Epoch  1  Loss  7795.6394361015
# accuracy: 88 %
# Epoch  2  Loss  7476.434200631706
# accuracy: 90 %
# Epoch  3  Loss  7429.284371277976
# accuracy: 91 %
# Epoch  4  Loss  7399.794071721951
# accuracy: 92 %
# Epoch  5  Loss  7374.095147988057
# accuracy: 93 %
# Epoch  6  Loss  7360.183345735367
# accuracy: 93 %
# Epoch  7  Loss  7343.454786724019
# accuracy: 93 %
# Epoch  8  Loss  7331.558122437573
# accuracy: 93 %
# Epoch  9  Loss  7319.605682273396
# accuracy: 94 %
# Epoch  10  Loss  7310.955456221739
# accuracy: 94 %
# Epoch  11  Loss  7300.144362474227
# accuracy: 94 %
# Epoch  12  Loss  7292.083888504974
# accuracy: 94 %
# Epoch  13  Loss  7282.3227097644485
# accuracy: 94 %
# Epoch  14  Loss  7276.746759878715
# accuracy: 94 %
# Epoch  15  Loss  7267.70032061774
# accuracy: 95 %
# Epoch  16  Loss  7261.3901898210925
# accuracy: 95 %
# Epoch  17  Loss  7254.557094400669
# accuracy: 95 %
# Epoch  18  Loss  7248.980283724729
# accuracy: 95 %
# Epoch  19  Loss  7242.114744297274
# accuracy: 95 %
# Epoch  20  Loss  7235.464850984685
# accuracy: 95 %
# Epoch  21  Loss  7229.462998536452
# accuracy: 95 %
# Epoch  22  Loss  7224.141231817587
# accuracy: 95 %
# Epoch  23  Loss  7217.477503020286
# accuracy: 95 %
# Epoch  24  Loss  7212.017617628423
# accuracy: 95 %
# Epoch  25  Loss  7206.526013750982
# accuracy: 95 %
# Epoch  26  Loss  7200.673278092035
# accuracy: 95 %
# Epoch  27  Loss  7195.646281444009
# accuracy: 96 %
# Epoch  28  Loss  7189.906175079672
# accuracy: 96 %
# Epoch  29  Loss  7185.586503403401
# accuracy: 96 %
# Epoch  30  Loss  7179.328694003868
# accuracy: 96 %
# Epoch  31  Loss  7174.049581344334
# accuracy: 96 %
# accuracy: 96 %
