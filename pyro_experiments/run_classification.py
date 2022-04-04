from torchvision.datasets import MNIST
from torchvision import transforms as T

from src.models.mle_classify import MLEClassify
from src.models.deep_mle_classify import DeepMLEClassify
from src.models.conv_classify import ConvClassify
from src.models.bnn_classification import BNNClassification
from src.commons.pyro_training import train_classification, prepare_loaders
from src.commons.utils import d

classify_models = {"mle_classify": MLEClassify, "deep_mle_classify": DeepMLEClassify, "conv_classify": ConvClassify}


def run_training(x, y, activation, net_model, net_args, model_args, epochs, b_size):
    train_loader, test_loader = prepare_loaders(x, y, b_size)
    net_args["activation"] = activation
    net = classify_models[net_model](**net_args)
    model_args["model"] = net
    model = BNNClassification(**model_args)
    train_classification(train_loader, test_loader, model, epochs)


# exapmle
train_set = MNIST(
    "datasets", train=True, transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]), download=True
)
test_set = MNIST(
    "datasets", train=False, transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]), download=True
)
net_args = {"in_size": 28 * 28, "hidden_size": 128, "out_size": 10}
model_args = {"mean": d(0.0), "std": d(100.0)}

run_training(
    x=train_set,
    y=test_set,
    activation="leaky_relu",
    net_model="mle_classify",
    net_args=net_args,
    model_args=model_args,
    epochs=150,
    b_size=512,
)
