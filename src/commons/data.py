import numpy as np
import torch as t
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms as A
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST


def get_dataset(path: str, dataset: Dataset, transform) -> Dataset:
    return ConcatDataset(
        [
            dataset(path, train=True, download=True, transform=transform),
            dataset(path, train=False, download=True, transform=transform),
        ]
    )


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str,
        dataset_path: str,
        train_batch_size: int,
        valtest_batch_size: int,
        in_channels: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.train_batch_size = train_batch_size
        self.valtest_batch_size = valtest_batch_size
        self.normalization = (
            ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if in_channels != 1 else ((0.1307,), (0.3081,))
        )
        self.train_transform = A.Compose(
            [
                A.ToTensor(),
                A.Normalize(*self.normalization),
                A.RandomAffine(degrees=(0, 70), translate=(0.1, 0.3), scale=(0.8, 1.2)),
            ]
        )
        self.val_test_transform = A.Compose([A.ToTensor(), A.Normalize(*self.normalization)])
        self.splits = {"train": 0.7, "val": 0.15, "test": 0.15}
        self.num_workers = num_workers

    def setup(self, stage: str):

        # TODO introduce parser options train-valid, valid-train, random_ratio_train/valid(float)
        # Get rid of test

        if self.dataset == "MNIST":
            self.train_dataset = get_dataset(self.dataset_path, MNIST, self.train_transform)
            self.val_test_dataset = get_dataset(self.dataset_path, MNIST, self.val_test_transform)
        elif self.dataset == "FashionMNIST":
            self.train_dataset = get_dataset(self.dataset_path, FashionMNIST, self.train_transform)
            self.val_test_dataset = get_dataset(self.dataset_path, FashionMNIST, self.val_test_transform)
        elif self.dataset == "CIFAR10":
            self.train_dataset = get_dataset(self.dataset_path, CIFAR10, self.train_transform)
            self.val_test_dataset = get_dataset(self.dataset_path, CIFAR10, self.val_test_transform)
        else:
            raise NotImplementedError(f"{self.dataset} is currently unsupported")

        indices = np.arange(len(self.val_test_dataset))
        np.random.shuffle(indices)

        select_index = lambda ratio: int(len(indices) * ratio)
        if stage != "test":
            self.splits = {
                "train": SubsetRandomSampler(indices[: select_index(self.splits["train"])]),
                "val": SubsetRandomSampler(
                    indices[
                        select_index(self.splits["train"]) : select_index(self.splits["train"] + self.splits["val"])
                    ]
                ),
                "test": SubsetRandomSampler(indices[select_index(self.splits["train"] + self.splits["val"]) :]),
            }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.train_batch_size, sampler=self.splits["train"], num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_test_dataset, self.valtest_batch_size, sampler=self.splits["val"], num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_test_dataset, self.valtest_batch_size, sampler=self.splits["test"], num_workers=self.num_workers
        )
