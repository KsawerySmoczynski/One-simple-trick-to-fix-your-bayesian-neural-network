import torch
from torchvision.datasets import MNIST
from torchvision import transforms as T

from src.models.mle_classify import MLEClassify
from src.models.deep_mle_classify import DeepMLEClassify
from src.models.conv_classify import ConvClassify
from src.models.mle_classification import MLEClassification
from src.models.bnn_classification import BNNClassification
from src.commons.pyro_training import train_classification, prepare_loaders
from src.commons.utils import d, device

classify_models = {"mle_classify": MLEClassify, "deep_mle_classify": DeepMLEClassify, "conv_classify": ConvClassify}


def run_training(x, y, activation, net_model, net_args, model_args, epochs, b_size):
    train_loader, test_loader = prepare_loaders(x, y, b_size)
    net_args["activation"] = activation
    net = classify_models[net_model](**net_args)
    model_args["model"] = net
    model = BNNClassification(**model_args)
    return train_classification(train_loader, test_loader, model, epochs)


# exapmle
train_set = MNIST(
    "datasets", train=True, transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]), download=True
)
test_set = MNIST(
    "datasets", train=False, transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]), download=True
)
net_args = {"in_size": 28 * 28, "hidden_size": 128, "out_size": 10}
model_args = {"mean": d(0.0), "std": d(100.0)}

model, guide = run_training(
    x=train_set,
    y=test_set,
    activation="leaky_relu",
    net_model="mle_classify",
    net_args=net_args,
    model_args=model_args,
    epochs=40,
    b_size=512,
)


# Convert Bayesian model back to standard Neural Network
net = MLEClassification(model=model, guide=guide, net=model.net)

# Check that the obtained model works
from torch.utils.data import DataLoader

test_loader = DataLoader(
        dataset=test_set, batch_size=512, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2
    )

net.eval()
ok = 0
for X, y in test_loader:
    X = X.to(device)
    y = y.to(device)
    out = net(X)
    ok += (y == torch.max(out, dim=1)[1]).sum()

print((ok/ len(test_loader.dataset)).item())

torch.save({'state_dict': net.net.state_dict()}, '../scripts/params/mle')
