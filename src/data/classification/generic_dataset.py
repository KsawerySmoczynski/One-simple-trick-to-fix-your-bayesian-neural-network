import torch
from sklearn.datasets import fetch_california_housing
from torch.utils.data import Dataset


class GenericDataset(Dataset):
    def __init__(self, data, targets):
        super().__init__()
        self.data = data
        self.targets = targets
        self.data = torch.tensor(self.data).float()
        # self.targets = torch.tensor(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.targets[idx]
        return X, y
