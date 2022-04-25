import torch
from sklearn.datasets import fetch_california_housing
from torch.utils.data import Dataset
from torchvision import transforms


class CaliforniaHousingDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, train: bool = True, ratio: float = 0.8):
        super().__init__()
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

        # TODO Apply normalization transforms with values from sample on train and from population on test
        self.transform = transform or (lambda x: ((x - self.data_mean) / self.data_std))
        self.target_transform = target_transform or (lambda x: ((x - self.targets_mean) / self.targets_std))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.transform(self.data[idx])
        y = self.target_transform(self.targets.data[idx])
        return X, y
