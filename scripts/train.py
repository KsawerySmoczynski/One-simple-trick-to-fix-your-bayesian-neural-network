from src.commons.cli import BayesianCLI
from src.commons.data import DataModule
from src.commons.module import BayesianModule

cli = BayesianCLI(model_class=BayesianModule, datamodule_class=DataModule, save_config_overwrite=True)
