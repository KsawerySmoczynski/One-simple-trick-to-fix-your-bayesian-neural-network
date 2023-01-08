# from https://gist.github.com/martinferianc/db7615c85d5a3a71242b4916ea6a14a2

import numpy as np
import torch
import torchvision.datasets as datasets
import logging
import os
from os import path
from sklearn.model_selection import KFold
import pandas as pd
import zipfile
import urllib.request

from torch.utils.data import Dataset
from src.data.datamodule import DataModule
from torchvision import transforms
from typing import Dict
import math

class UCIDatasets():
    def __init__(self,  name, data_path=""):
        self.datasets = {
            "housing": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
            "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
            "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
            "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"}
        self.data_path = data_path
        self.name = name
        self._load_dataset()
   
    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not path.exists(self.data_path+"UCI"):
            os.mkdir(self.data_path+"UCI")

        url = self.datasets[self.name]
        file_name = url.split('/')[-1]
        if not path.exists(self.data_path+"UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path+"UCI/" + file_name)
        data = None


        if self.name == "housing":
            data = pd.read_csv(self.data_path+'UCI/housing.data',
                        header=0, delimiter="\s+").values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "concrete":
            data = pd.read_excel(self.data_path+'UCI/Concrete_Data.xls',
                               header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "energy":
            data = pd.read_excel(self.data_path+'UCI/ENB2012_data.xlsx',
                                 header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "power":
            zipfile.ZipFile(self.data_path +"UCI/CCPP.zip").extractall(self.data_path +"UCI/CCPP/")
            data = pd.read_excel(self.data_path+'UCI/CCPP/CCPP/Folds5x2_pp.xlsx', header=0).values
            np.random.shuffle(data)
            self.data = data
        elif self.name == "wine":
            data = pd.read_csv(self.data_path + 'UCI/winequality-red.csv',
                               header=1, delimiter=';').values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "yacht":
            data = pd.read_csv(self.data_path + 'UCI/yacht_hydrodynamics.data',
                               header=1, delimiter='\s+').values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        self.samples = data[:, :-2]
        self.targets = data[:, -1]


class UCIDataset(Dataset):
    def __init__(self, name, train: bool = True, ratio: float = 0.8):
        super().__init__()

        dataset = UCIDatasets(name)

        if not train:
            ratio = 1 - ratio

        if train:
            self.data = dataset.samples[:int(ratio * dataset.samples.shape[0])]
            self.targets = dataset.targets[:int(ratio * dataset.targets.shape[0])]

        else:
            self.data = dataset.samples[int(ratio * dataset.samples.shape[0]):]
            self.targets = dataset.targets[int(ratio * dataset.targets.shape[0]):]

        self.data = torch.Tensor(self.data).float()
        self.targets = torch.Tensor(self.targets).float()

        self.data_mean = self.data.mean(dim=0)
        self.data_std = self.data.std(dim=0)

        self.targets_mean = self.targets.mean(dim=0) + 0.001
        self.targets_std = self.targets.std(dim=0) + 0.001

        # print(self.data)

        print ("Data shape")
        print(self.data.shape)
        print(self.targets.shape)

        # print(np.isfinite(self.data).all())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = (self.data[idx] - self.data_mean) / self.data_std
        y = (self.targets[idx] - self.targets_mean) / self.targets_std
        return X, y

class UCI(DataModule):
    def __init__(
        self,
        name,
        train_batch_size: int,
        test_batch_size: int,
        root: str = "",
        train_ratio: float = 0.8,
        validation_ratio: float = 0.2,
        dataloader_args: Dict = {},
    ):
        super().__init__(train_batch_size, test_batch_size, root, train_ratio, validation_ratio, 0, dataloader_args)
        
        self.train_dataset = UCIDataset(train = True, name = name, ratio = train_ratio)
        self.test_dataset = UCIDataset(train = False, name= name, ratio = validation_ratio)

        sse = 0
        for _, y in self.test_dataset:
            sse += y**2

        rmse = math.sqrt(sse/len(self.test_dataset))

        print(f"NAIVE RMSE: {rmse}")

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
