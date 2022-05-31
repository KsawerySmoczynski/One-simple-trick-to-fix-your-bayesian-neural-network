from pytorch_lightning import LightningDataModule

from src.data import DataModule


class DataModule(LightningDataModule):
    def __init__(self, dataset: DataModule, n_classes: int):
        super().__init__()
        self.dataset = dataset
        self.n_classes = n_classes

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return self.dataset.train_dataloader()

    def val_dataloader(self):
        return self.dataset.validation_dataloader()

    def test_dataloader(self):
        return self.dataset.test_dataloader()
