import torch as t
import pyro
pyro.set_rng_seed(101)

# Normal dist
mean = 0
std = 1
normal = t.distributions.Normal(mean, std)
x = normal.rsample()
print("sample", x)
print("log prob", normal.log_prob(x))


def weather():
    weather = t.distributions.Bernoulli(0.3).sample()
    weather = "cloudy" if weather else "sunny"  
    return weather, t.distributions.Normal(55 if weather=="cloudy" else 75, 10 if weather == "cloudy" else 15).item()


def pyro_weather():
    weather = pyro.sample("cloudy", pyro.distributions.Bernoulli(0.3))
    weather = "cloudy" if weather else "sunny"  
    return weather, pyro.sample("my_sample", pyro.distributions.Normal(55 if weather=="cloudy" else 75, 10 if weather == "cloudy" else 15)).item()
