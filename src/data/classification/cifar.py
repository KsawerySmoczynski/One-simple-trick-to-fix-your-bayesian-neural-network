from typing import Dict

from torchvision import transforms
from torchvision.datasets import CIFAR10 as CIFAR10Dataset

from src.data.datamodule import DataModule


class CIFAR10(DataModule):
    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        root: str = "datasets",
        train_ratio: float = 0.7,
        validation_ratio: float = 0.3,
        dataloader_args: Dict = {},
    ):
        super().__init__(train_batch_size, test_batch_size, root, train_ratio, validation_ratio, 0, dataloader_args)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        )
        self.train_dataset = CIFAR10Dataset(root=self.root, train=True, transform=transform, download=True)
        self.test_dataset = CIFAR10Dataset(root=self.root, train=False, transform=transform, download=True)
        # TODO this is only for tests
        # self.train_dataset.data = self.train_dataset.data[:640]
        # self.train_dataset.targets = self.train_dataset.targets[:640]
        # self.test_dataset.data = self.test_dataset.data[:256]
        # self.test_dataset.targets = self.test_dataset.targets[:256]

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
