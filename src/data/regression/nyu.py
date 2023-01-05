import torch

from src.data.regression.nyu_parser import NYUv2Dataset
from src.data.datamodule import DataModule
from torchvision import transforms
from typing import Dict


class NYUv2(DataModule):
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
        
        t = transforms.Compose([transforms.ToTensor()])

        self.train_dataset = NYUv2Dataset(root="data", train=True, download=True, rgb_transform=t, seg_transform=None, sn_transform=None, depth_transform=t)
        self.test_dataset = NYUv2Dataset(root="data", train=False, download=True, rgb_transform=t, seg_transform=None, sn_transform=None, depth_transform=t)

        train_ratio = len(self.train_dataset) / (len(self.train_dataset) + len(self.test_dataset))
        validation_ratio = 1 - train_ratio

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
