from src.commons.cli import CLI
from src.commons.data import DataModule
from src.commons.module import TrainingModule

cli = CLI(model_class=TrainingModule, datamodule_class=DataModule, save_config_overwrite=True)
