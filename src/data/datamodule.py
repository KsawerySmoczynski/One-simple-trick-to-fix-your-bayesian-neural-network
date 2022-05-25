import abc
import multiprocessing
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler


class AbstractDataModule(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "train_dataloader")
            and callable(subclass.train_dataloader)
            and hasattr(subclass, "validation_dataloader")
            and callable(subclass.validation_dataloader)
            and hasattr(subclass, "test_dataloader")
            and callable(subclass.test_dataloader)
            or NotImplemented
        )

    @abc.abstractmethod
    def train_dataloader(self):
        raise NotImplementedError

    @abc.abstractmethod
    def validation_dataloader(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_dataloader(self):
        raise NotImplementedError


class DataModule(AbstractDataModule):
    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        root: str = "datasets",
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        dataloader_args: Dict = {},
    ):
        super().__init__()
        assert train_ratio + validation_ratio + test_ratio == 1.0, "Train/validation/test ratios should sum up to 1"
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.root = root
        self.train_sampler = None
        self.validation_sampler = None
        self.test_sampler = None
        self.train_dataloader_config = None
        self.validation_dataloader_config = None
        self.test_dataloader_config = None
        self.dataloader_args = dataloader_args

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def _get_train_val_test_samplers(n: int, train_ratio: float, validation_ratio: float, test_ratio: float = None):
        indices = np.arange(n)
        train_indices = np.random.choice(indices, size=int(n * train_ratio), replace=False)
        validation_indices = np.random.choice(
            indices[np.in1d(indices, train_indices, invert=True)], size=int(n * validation_ratio), replace=False
        )
        if test_ratio:
            test_indices = indices[np.in1d(indices, np.concatenate([train_indices, validation_indices]), invert=True)]
            return (
                SubsetRandomSampler(train_indices),
                SubsetRandomSampler(validation_indices),
                SubsetRandomSampler(test_indices),
            )
        else:
            return SubsetRandomSampler(train_indices), SubsetRandomSampler(validation_indices)

    @staticmethod
    def _get_dataloader_configuration(
        dataset, batch_size: int, sampler: Sampler = None, shuffle: bool = None, dataloader_args: Dict = {}
    ):
        assert bool(sampler) ^ isinstance(shuffle, bool), "Either provide sampler or shuffle argument"

        dataloader_args = {"dataset": dataset, "batch_size": batch_size, **dataloader_args}
        if bool(sampler):
            dataloader_args = {**dataloader_args, "sampler": sampler}
        else:
            dataloader_args = {**dataloader_args, "shuffle": shuffle}

        return dataloader_args

    def train_dataloader(self, overrides: Dict = None):
        return self._return_dataloader(self.train_dataloader_config, overrides)

    def validation_dataloader(self, overrides: Dict = None):
        return self._return_dataloader(self.validation_dataloader_config, overrides)

    def test_dataloader(self, overrides: Dict = None):
        return self._return_dataloader(self.test_dataloader_config, overrides)

    @staticmethod
    def _return_dataloader(config: Dict, overrides: Dict = None):
        if overrides:
            config = {**config, **overrides}
        return DataLoader(**config)
