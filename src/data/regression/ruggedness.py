import torch
from sklearn.datasets import fetch_california_housing
from torch.utils.data import Dataset

from src.data.datamodule import DataModule
from torchvision import transforms
from typing import Dict
import math
import pandas as pd
import numpy as np

class RuggednessDataset(Dataset):
    def __init__(self):
        super().__init__()

        DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
        data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
        df = data[["cont_africa", "rugged", "rgdppc_2000"]]
        df = df[np.isfinite(df.rgdppc_2000)]
        df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

        df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]
        data = torch.tensor(df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].values,
                        dtype=torch.float)

        self.data, self.targets = data[:, :-1], data[:, -1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class Ruggedness(DataModule):
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
        
        self.dataset = RuggednessDataset()

        batch_size = len(self.dataset)

        sse = 0
        for _, y in self.dataset:
            sse += y**2

        rmse = math.sqrt(sse/len(self.dataset))

        print(f"NAIVE RMSE: {rmse}")
        print(f"Dataset len: {len(self.dataset)}")

        self.train_sampler, _ = self._get_train_val_test_samplers(
            len(self.dataset), 1, 0
        )

        self.validation_sampler = self.train_sampler

        self.train_dataloader_config = self._get_dataloader_configuration(
            self.dataset, batch_size, self.train_sampler, dataloader_args=self.dataloader_args
        )
        self.validation_dataloader_config = self._get_dataloader_configuration(
            self.dataset, batch_size, self.train_sampler, dataloader_args=self.dataloader_args
        )
        self.test_dataloader_config = self._get_dataloader_configuration(
            self.dataset, batch_size, shuffle=False, dataloader_args=self.dataloader_args
        )
