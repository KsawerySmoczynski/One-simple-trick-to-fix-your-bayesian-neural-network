from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

from src.datasets.california import CaliforniaHousingDataset


def get_dataloaders(train_dataset, test_dataset, train_batch_size: int, num_workers: int, *args, **kwargs):
    train_dataloader = DataLoader(train_dataset, train_batch_size, num_workers)
    test_dataloader = DataLoader(test_dataset, train_batch_size, num_workers)
    return train_dataloader, test_dataloader


def get_datasets(dataset: str, dataset_path: str, train_transform, test_transform, *args, **kwargs):
    if dataset == "MNIST":
        train_dataset = MNIST(dataset_path, train=True, download=False, transform=train_transform)
        test_dataset = MNIST(dataset_path, train=False, download=False, transform=test_transform)
    elif dataset == "FashionMNIST":
        train_dataset = FashionMNIST(dataset_path, train=True, download=False, transform=train_transform)
        test_dataset = FashionMNIST(dataset_path, train=False, download=False, transform=test_transform)
    elif dataset == "CaliforniaHousingDataset":
        train_dataset = CaliforniaHousingDataset(train=True)
        test_dataset = CaliforniaHousingDataset(train=False)
    else:
        raise NotImplementedError(f"{dataset} is currently unsupported")

    return train_dataset, test_dataset


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str,
        dataset_path: str,
        train_batch_size: int,
        test_batch_size: int,
        in_channels: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.normalization = (
            ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if in_channels == 3 else ((0.1307,), (0.3081,))
        )
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*self.normalization),
                transforms.RandomAffine(degrees=(0, 70), translate=(0.1, 0.3), scale=(0.8, 1.2)),
            ]
        )
        self.test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*self.normalization)])
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_dataset, self.val_dataset = get_datasets(
            self.dataset,
            self.dataset_path,
            self.train_transform,
            self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, self.test_batch_size, num_workers=self.num_workers)
