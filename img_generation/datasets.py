from dataclasses import dataclass
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset


class DatasetType(Enum):
    TINY_IMAGNET = 0
    CIFAR10 = 1
    CIFAR100 = 2


@dataclass
class Config:
    dataset_type: DatasetType = DatasetType.CIFAR10
    batch_size: int = 128
    num_workers: int = 4
    data_dir: str


class ImageDatasetHF(Dataset):
    def __init__(self, pairs, transform=None, target_transform=None, classes=[]):
        super().__init__()
        self.pairs = pairs
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image, label = self.pairs[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


def load_datasets(config: Config):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )  # possible further data augmentation - if we decide so

    if config.dataset_type.name is DatasetType.TINY_IMAGNET:
        dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir=Config.data_dir)
        # splits = ['train', 'valid']
        # dataset item = {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=64x64 at 0x116907380>, 'label': 0}

        train_data = [(pair["image"], pair["label"]) for pair in dataset["train"]]
        test_data = [(pair["image"], pair["label"]) for pair in dataset["valid"]]
        labels = dataset["train"].features["label"].names

        train_set = ImageDatasetHF(train_data, transform=transform, classes=labels)
        test_set = ImageDatasetHF(test_data, transform=transform, classes=labels)

    elif config.dataset_type.name is DatasetType.CIFAR10:
        train_set = datasets.CIFAR10(
            root=Config.data_dir, train=True, transform=transform, download=True
        )
        test_set = datasets.CIFAR10(
            root=Config.data_dir, train=False, transform=transform, download=True
        )
    else:
        train_set = datasets.CIFAR100(
            root=Config.data_dir, train=True, transform=transform, download=True
        )
        test_set = datasets.CIFAR100(
            root=Config.data_dir, train=False, transform=transform, download=True
        )

    train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])

    print(
        f"Loaded train dataset: {len(train_set)} images, {len(train_set.classes)} classes."
    )

    return train_set, val_set, test_set


def prep_dataloaders(config: Config):
    train_set, val_set, test_set = load_datasets(config)

    dataloaders = {
        "train": DataLoader(
            train_set,
            batch_size=Config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=Config.num_workers,
        ),
        "val": DataLoader(
            val_set,
            batch_size=Config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=Config.num_workers,
        ),
        "test": DataLoader(
            test_set,
            batch_size=Config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=Config.num_workers,
        ),
    }

    return dataloaders
