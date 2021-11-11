from src.commons.module import BayesianModule
from src.commons.data import DataModule
from src.commons.cli import BayesianCLI

BayesianCLI(model_class=BayesianModule, datamodule_class=DataModule)