# %reset -s -f

import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import seaborn as sns
import torch
from pyro.nn import PyroModule
from torch import nn

# for CI testing
smoke_test = "CI" in os.environ
assert pyro.__version__.startswith("1.7.0")
pyro.set_rng_seed(1)

plt.ion()

# Set matplotlib settings
# %matplotlib
plt.style.use("default")

# GET DATA
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"].hist()
plt.show()
plt.close()

df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
df["rgdppc_2000"].hist()
plt.show()
plt.close()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = df[df["cont_africa"] == 1]
non_african_nations = df[df["cont_africa"] == 0]
sns.scatterplot(non_african_nations["rugged"], non_african_nations["rgdppc_2000"], ax=ax[0])
ax[0].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="Non African Nations")
sns.scatterplot(african_nations["rugged"], african_nations["rgdppc_2000"], ax=ax[1])
ax[1].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations")
plt.close()


df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]
data = torch.tensor(df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].values, dtype=torch.float)

x_data, y_data = data[:, :-1], data[:, -1]
linear_reg_model = PyroModule[nn.Linear](3, 1)
optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)
loss_fn = nn.MSELoss(reduction="sum")
n_iter = 1500 * 8


def train_step():
    optim.zero_grad()
    out = linear_reg_model(x_data).squeeze(-1)
    loss = loss_fn(out, y_data)
    loss.backward()
    optim.step()

    return loss


linear_reg_model = linear_reg_model.to("cuda")
y_data = y_data.cuda()
x_data = x_data.cuda()


for i in range(n_iter):
    loss = train_step()
    if i % 100 == 0:
        print(f"Loss {loss.cpu().item():.4f}")

linear_reg_model = linear_reg_model.to("cpu")

for name, param in linear_reg_model.named_parameters():
    print(name, param.data.numpy())


fit = df.copy()
fit["mean"] = linear_reg_model(x_data.cpu()).detach().numpy()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = fit[fit["cont_africa"] == 1]
non_african_nations = fit[fit["cont_africa"] == 0]
fig.suptitle("Regression Fit", fontsize=16)
ax[0].plot(non_african_nations["rugged"], non_african_nations["rgdppc_2000"], "o")
ax[0].plot(non_african_nations["rugged"], non_african_nations["mean"], linewidth=2)
ax[0].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="Non African Nations")
ax[1].plot(african_nations["rugged"], african_nations["rgdppc_2000"], "o")
ax[1].plot(african_nations["rugged"], african_nations["mean"], linewidth=2)
ax[1].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations")
plt.close()

# Bayesian linear regression with pyro
from pyro.nn import PyroSample


class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0.0, 10.0).expand([out_features, in_features]).to_event(1))
        self.linear.bias = PyroSample(dist.Normal(0.0, 100.0).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


from pyro.infer.autoguide import AutoDiagonalNormal

model = BayesianRegression(3, 1)
guide = AutoDiagonalNormal(model)

from pyro.infer import SVI, Trace_ELBO

optim = pyro.optim.Adam({"lr": 0.03})
DEVICE = "cpu"
guide = guide.to(DEVICE)
model = model.to(DEVICE)
x_data = x_data.to(DEVICE)
y_data = y_data.to(DEVICE)

svi = SVI(model, guide, optim, loss=Trace_ELBO())

pyro.clear_param_store()


for j in range(n_iter):
    loss = svi.step(x_data, y_data)
    if j % 100 == 0:
        print(f"I{j+1}, loss: {loss/len(data):.4f}")

guide.requires_grad_(False)


for name, param in pyro.get_param_store().items():
    print(name, pyro.param(name))

# Plot estimates
from pyro.infer import Predictive


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats


predictive = Predictive(model, guide=guide, num_samples=800, return_sites=("linear.weight", "obs", "_RETURN"))
samples = predictive(x_data)
pred_summary = summary(samples)

pred_summary["_RETURN"]["mean"]
pred_summary["_RETURN"]["std"]

mu = pred_summary["_RETURN"]
y = pred_summary["obs"]

predictions = pd.DataFrame(
    {
        "cont_africa": x_data[:, 0],
        "rugged": x_data[:, 1],
        "mu_mean": mu["mean"],
        "mu_perc_5": mu["5%"],
        "mu_perc_95": mu["95%"],
        "y_mean": y["mean"],
        "y_perc_5": y["5%"],
        "y_perc_95": y["95%"],
        "true_gdp": y_data,
    }
)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = predictions[predictions["cont_africa"] == 1]
non_african_nations = predictions[predictions["cont_africa"] == 0]
african_nations = african_nations.sort_values(by=["rugged"])
non_african_nations = non_african_nations.sort_values(by=["rugged"])
fig.suptitle("Regression line 90% CI", fontsize=16)
ax[0].plot(non_african_nations["rugged"], non_african_nations["mu_mean"])
ax[0].fill_between(
    non_african_nations["rugged"], non_african_nations["mu_perc_5"], non_african_nations["mu_perc_95"], alpha=0.5
)
ax[0].plot(non_african_nations["rugged"], non_african_nations["true_gdp"], "o")
ax[0].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="Non African Nations")
idx = np.argsort(african_nations["rugged"])
ax[1].plot(african_nations["rugged"], african_nations["mu_mean"])
ax[1].fill_between(african_nations["rugged"], african_nations["mu_perc_5"], african_nations["mu_perc_95"], alpha=0.5)
ax[1].plot(african_nations["rugged"], african_nations["true_gdp"], "o")
ax[1].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations")
plt.close()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)
ax[0].plot(non_african_nations["rugged"], non_african_nations["y_mean"])
ax[0].fill_between(
    non_african_nations["rugged"], non_african_nations["y_perc_5"], non_african_nations["y_perc_95"], alpha=0.5
)
ax[0].plot(non_african_nations["rugged"], non_african_nations["true_gdp"], "o")
ax[0].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="Non African Nations")
idx = np.argsort(african_nations["rugged"])

ax[1].plot(african_nations["rugged"], african_nations["y_mean"])
ax[1].fill_between(african_nations["rugged"], african_nations["y_perc_5"], african_nations["y_perc_95"], alpha=0.5)
ax[1].plot(african_nations["rugged"], african_nations["true_gdp"], "o")
ax[1].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations")
