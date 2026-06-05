from dataclasses import dataclass
from enum import Enum
import pickle
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset


def discretize(sample):
    return (sample * 255).to(torch.long)


PIXEL_CNN_TRANSFORM = transforms.Compose([transforms.ToTensor(), discretize])

VAE_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

PIXEL_CNN_HF_TRANSORM = transforms.Compose(
    [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((32, 32)),
        *PIXEL_CNN_TRANSFORM.transforms,
    ]
)

VAE_HF_TRANSORM = transforms.Compose(
    [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((32, 32)),
        *VAE_TRANSFORM.transforms,
    ]
)  # tiny-imagenet dataset is of poor quality, can occur monochromatic photo; resize to match cifar size


class DatasetType(Enum):
    TINY_IMAGNET = 0
    CIFAR10 = 1
    CIFAR100 = 2


@dataclass
class Config:
    dataset_type: DatasetType
    data_dir: str
    batch_size: int = 128
    num_workers: int = 4


class ImageDatasetHF(Dataset):
    def __init__(self, pairs, transform=None, target_transform=None):
        super().__init__()
        self.pairs = pairs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image, label = self.pairs[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


def _pickle_data(data: tuple, config: Config):
    train_set, val_set, test_set, classes = data

    with open(os.path.join([config.data_dir, "train.pkl"]), "rb") as fh:
        pickle.dump(train_set, fh)
    with open(os.path.join([config.data_dir, "val.pkl"]), "rb") as fh:
        pickle.load(val_set, fh)
    with open(os.path.join([config.data_dir, "test.pkl"]), "rb") as fh:
        pickle.load(test_set, fh)
    with open(os.path.join([config.data_dir, "classes.pkl"]), "rb") as fh:
        pickle.load(classes, fh)


def _unpickle_data(config: Config):
    with open(os.path.join([config.data_dir, "train.pkl"]), "rb") as fh:
        train_set = pickle.load(fh)
    with open(os.path.join([config.data_dir, "val.pkl"]), "rb") as fh:
        val_set = pickle.load(fh)
    with open(os.path.join([config.data_dir, "test.pkl"]), "rb") as fh:
        test_set = pickle.load(fh)
    with open(os.path.join([config.data_dir, "classes.pkl"]), "rb") as fh:
        classes = pickle.load(fh)

    return train_set, val_set, test_set, classes


def load_datasets(config: Config, transforms: transforms.Compose):
    """
    Tiny-imagenet:
        - splits = ['train', 'valid']
        - dataset item = {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=64x64 at 0x116907380>, 'label': 0}
    Cifar:
        - PIL image mode=RGB size=32x32
    """

    is_dataset_prepared = all(
        [
            os.path.exists(),
            os.path.exists(os.path.join([config.data_dir, "val.pkl"])),
            os.path.exists(os.path.join([config.data_dir, "test.pkl"])),
        ]
    )

    if not is_dataset_prepared:
        if config.dataset_type is DatasetType.TINY_IMAGNET:
            dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir=config.data_dir)

            train_data = [(pair["image"], pair["label"]) for pair in dataset["train"]]
            test_data = [(pair["image"], pair["label"]) for pair in dataset["valid"]]
            classes = dataset["train"].features["label"].names

            train_set = ImageDatasetHF(train_data, transform=transforms)
            test_set = ImageDatasetHF(test_data, transform=transforms)

        elif config.dataset_type is DatasetType.CIFAR10:
            train_set = datasets.CIFAR10(root=config.data_dir, train=True, transform=transforms, download=True)
            test_set = datasets.CIFAR10(root=config.data_dir, train=False, transform=transforms, download=True)
            classes = train_set.classes
        else:
            train_set = datasets.CIFAR100(root=config.data_dir, train=True, transform=transforms, download=True)
            test_set = datasets.CIFAR100(root=config.data_dir, train=False, transform=transforms, download=True)
            classes = train_set.classes

        train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])
        _pickle_data((train_set, val_set, test_set, classes), config)

    train_set, val_set, test_set, classes = _unpickle_data(config)

    print(f"Loaded train dataset: {len(train_set)} images, {len(classes)} classes.")

    return train_set, val_set, test_set, classes


def load_dataloaders(config: Config, transforms: transforms.Compose):
    train_set, val_set, test_set, classes = load_datasets(config, transforms)

    dataloaders = {
        "train": DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=config.num_workers,
        ),
        "val": DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
        ),
        "test": DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
        ),
    }

    return dataloaders, classes
