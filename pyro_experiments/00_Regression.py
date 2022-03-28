import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn.module import PyroModule, PyroSample
from sklearn import datasets
from torch import nn

steps = 400
log_steps = 25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x, y = datasets.load_diabetes(return_X_y=True, as_frame=False)

x = torch.Tensor(x)
y = torch.Tensor(y)
x = x.to(device)
y = y.to(device)


def d(a):
    return torch.Tensor([a]).to(device)


class SimpleNN(PyroModule):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layer1 = PyroModule[nn.Linear](in_size, hidden_size)
        self.layer1.weight = PyroSample(dist.Normal(d(0.0), d(1.0)).expand([hidden_size, in_size]).to_event(2))
        self.layer1.bias = PyroSample(dist.Normal(d(0.0), d(10.0)).expand([hidden_size]).to_event(1))

        self.layer2 = PyroModule[nn.Linear](hidden_size, out_size)
        self.layer2.weight = PyroSample(dist.Normal(d(0.0), d(1.0)).expand([out_size, hidden_size]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(d(0.0), d(10.0)).expand([out_size]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(d(0.0), d(10.0)))
        h = self.layer1(x).squeeze(-1)
        mean = self.layer2(h).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


model = SimpleNN(10, 10, 1)
guide = AutoDiagonalNormal(model)

adam = pyro.optim.Adam({"lr": 0.03})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

pyro.clear_param_store()
for j in range(steps):
    loss = svi.step(x, y)
    if j % log_steps == 0:
        print("iteration: ", j)
        print("loss: ", loss / len(y))


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


from pyro.infer import Predictive

predictive = Predictive(model, guide=guide, num_samples=1000, return_sites=("model.layer2.weight", "obs", "_RETURN"))

samples = predictive(x[:1, ...].cuda())
output = model(x[:1, ...].cuda())
outputs = samples["obs"]
pred_summary = summary({k: v for k, v in samples.items() if k != "obs"})

import json

json.dump(pred_summary, open("summary.json", "w"), indent=4)
