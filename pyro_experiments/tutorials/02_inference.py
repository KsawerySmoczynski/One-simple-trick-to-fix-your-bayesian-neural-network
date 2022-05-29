import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro as p
import torch as t
from pyro import distributions as dist
from pyro import infer, optim

np.random.seed(101)
t.manual_seed(101)
p.set_rng_seed(101)


def scale(guess):
    weight = p.sample("weight", dist.Normal(guess, 1.0))
    return p.sample("measurement", dist.Normal(weight, 0.75)), weight


conditioned_scale = pyro.condition(scale, data={"measurement": t.tensor(9.5)})


def deferred_conditioned_scale(measurement, guess):
    return pyro.condition(scale, data={"measurement": measurement})(guess)


def perfect_guide(guess):
    loc = (0.75 ** 2 * guess + 9.5) / (1 + 0.75 ** 2)  # 9.14
    scale = np.sqrt(0.75 ** 2 / (1 + 0.75 ** 2))  # 0.6
    return pyro.sample("weight", dist.Normal(loc, scale))


scale(t.tensor(2.0))
conditioned_scale(t.tensor(2.0))
deferred_conditioned_scale(t.tensor(9.5), t.tensor(2.0))

guess = 8.5

pyro.clear_param_store()
svi = pyro.infer.SVI(
    model=conditioned_scale,
    guide=scale_parametrized_guide,
    optim=pyro.optim.Adam({"lr": 0.003}),
    loss=pyro.infer.Trace_ELBO(),
)


losses, a, b = [], [], []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step(guess))
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
print("a = ", pyro.param("a").item())
print("b = ", pyro.param("b").item())
