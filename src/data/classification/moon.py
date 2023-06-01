from typing import Dict

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from src.data.datamodule import DataModule
from src.data.classification.generic_dataset import GenericDataset
from sklearn.preprocessing import scale


class MOON(DataModule):
    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        root: str = "datasets",
        train_ratio: float = 0.5,
        validation_ratio: float = 0.5,
        dataloader_args: Dict = {},
    ):
        super().__init__(train_batch_size, test_batch_size, root, train_ratio, validation_ratio, 0, dataloader_args)
        
        X, Y =  make_moons(noise=0.2, random_state=0, n_samples=1000)
        X = scale(X)
        # Y = np.stack([Y, 1 - Y], axis=1)
        # print(Y.shape)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
        self.train_dataset = GenericDataset(X_train, Y_train) 
        self.test_dataset = GenericDataset(X_test, Y_test)
        
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
