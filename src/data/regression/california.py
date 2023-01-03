import torch
from sklearn.datasets import fetch_california_housing
from torch.utils.data import Dataset

from src.data.datamodule import DataModule
from torchvision import transforms
from typing import Dict

class CaliforniaHousingDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, train: bool = True, ratio: float = 0.8):
        super().__init__()
        ratio = ratio if train else 1 - ratio
        california_housing = fetch_california_housing()
        selection = int(len(california_housing["data"]) * ratio)
        self.data = california_housing["data"][:selection] if train else california_housing["data"][selection:]
        self.data = torch.tensor(self.data).float()
        self.targets = california_housing["target"][:selection] if train else california_housing["target"][selection:]
        self.targets = torch.tensor(self.targets).float()

        self.data_mean = self.data.mean(dim=0)
        self.data_std = self.data.std(dim=0)

        self.targets_mean = self.targets.mean(dim=0)
        self.targets_std = self.targets.std(dim=0)

        print(self.data.shape)
        print (f"NAIVE MODEL RMSE: {((self.targets_mean - self.targets)**2).sum().sqrt()}")

        # TODO Apply normalization transforms with values from sample on train and from population on test
        self.transform = transform or (lambda x: ((x - self.data_mean) / self.data_std))
        self.target_transform = target_transform or (lambda x: ((x - self.targets_mean) / self.targets_std))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.transform(self.data[idx])
        y = self.target_transform(self.targets.data[idx])
        return X, y

class CaliforniaHousing(DataModule):
    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        root: str = "",
        train_ratio: float = 0.7,
        validation_ratio: float = 0.3,
        dataloader_args: Dict = {},
    ):
        super().__init__(train_batch_size, test_batch_size, root, train_ratio, validation_ratio, 0, dataloader_args)
        
        self.train_dataset = CaliforniaHousingDataset(train = True, ratio = train_ratio)
        self.test_dataset = CaliforniaHousingDataset(train = False, ratio = validation_ratio)

        self.train_sampler, self.validation_sampler = self._get_train_val_test_samplers(
            len(self.train_dataset), train_ratio, validation_ratio
        )
        self.train_dataloader_config = self._get_dataloader_configuration(
            self.train_dataset, self.train_batch_size, self.train_sampler, dataloader_args=self.dataloader_args
        )
        self.validation_dataloader_config = self._get_dataloader_configuration(
            self.train_dataset, self.test_batch_size, self.validation_sampler, dataloader_args=self.dataloader_args
        )
        self.test_dataloader_config = self._get_dataloader_configuration(
            self.test_dataset, self.test_batch_size, shuffle=False, dataloader_args=self.dataloader_args
        )
