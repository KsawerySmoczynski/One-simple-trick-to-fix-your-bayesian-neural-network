import numpy as np
import scipy as sp
from scipy.stats import beta
from matplotlib import pyplot as plt
import torch as t
from torch.distributions import constraints
import pyro
from pyro import distributions as dist
from pyro.optim import Adam

beta = dist.Beta(300., 300.)
result = beta.sample([int(10e6)]).numpy()

# plt.hist(result, bins=100, density=True)
# plt.show()

pyro.clear_param_store()

def model(data):
    alpha0 = t.tensor(10.)
    beta0  = t.tensor(10.)
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # sequential
    # for i in pyro.plate("data_loop", len(data)): #plate is a context manager, each loop iteration is a different context
        # pyro.sample(f"obs_{i}", dist.Bernoulli(f), obs=data[i])
    #vectorized
    with pyro.plate("observe_data", size=1000, subsample_size=1) as ind:
        pyro.sample('obs', dist.Bernoulli(f), obs=data.index_select(0, ind))

def guide(data):
    alpha_q = pyro.param("alpha_q", t.tensor(15.), constraint=constraints.positive)
    beta_q = pyro.param("beta_q", t.tensor(15.), constraint=constraints.positive)
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))


adam_params = {"lr":5e-4, "betas":(.9, .999)}
optimizer = Adam(adam_params)

svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

data = t.zeros(1000)
data[:600] += 1
n_steps = 5_000

for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print(".")    

alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

# mean
inferred_mean = alpha_q / (alpha_q + beta_q)
# std
factor = beta_q / (alpha_q * (1. + alpha_q + beta_q))
inferred_std = inferred_mean * np.sqrt(factor)

print("\nbased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))