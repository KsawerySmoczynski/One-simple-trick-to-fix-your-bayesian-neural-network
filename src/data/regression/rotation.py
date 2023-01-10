import torch
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import Dataset

from src.data.datamodule import DataModule
from torchvision import transforms
from typing import Dict
import math


class RotationDataset(Dataset):
  def __init__(self, train, digits):
    super().__init__()
 
    self.mnist = MNIST(root='datasets', train=train, download=True)
    self.data = []
    self.targets = []
 
    for img, label in self.mnist:
      if label in digits:
        self.data.append(img)
        self.targets.append(label)
 
    self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
    
  def __len__(self):
      return len(self.targets)
 
  def __getitem__(self, idx):
    angle = np.random.random() * 90 - 45
    img1 = self.transform(self.data[idx])
    img2 = transforms.functional.rotate(img1, angle)
    
    return torch.concat([img1, img2]), angle / 45
 
class Rotation(DataModule):
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
        
        self.train_dataset = RotationDataset(train = True, digits=[1])
        self.test_dataset = RotationDataset(train = False, digits=[1])

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
