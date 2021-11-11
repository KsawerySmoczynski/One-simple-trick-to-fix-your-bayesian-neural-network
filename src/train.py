from pytorch_lightning.utilities.cli import LightningCLI
from module import BayesianModule
from data import DataModule

LightningCLI(model_class=BayesianModule, datamodule_class=DataModule)