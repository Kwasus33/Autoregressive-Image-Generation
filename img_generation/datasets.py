from dataclasses import dataclass
from enum import Enum
import json
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
    random_seed: int = 42


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


def _save_split_indices(indices: tuple, config: Config):
    with open(os.path.join([config.data_dir, "indices.json"]), "w") as fh:
        json.dump(indices, fh)


def _load_split_indices(config: Config):
    with open(os.path.join([config.data_dir, "indices.json"]), "r") as fh:
        indices = json.load(fh)
    return indices


def load_datasets(config: Config, transforms: transforms.Compose):
    """
    Tiny-imagenet:
        - splits = ['train', 'valid']
        - dataset item = {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=64x64 at 0x116907380>, 'label': 0}
    Cifar:
        - PIL image mode=RGB size=32x32
    """
    os.makedirs(config.data_dir, exist_ok=True)

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

    if not os.path.exists(os.path.join([config.data_dir, "indices.json"])):
        total_len = len(train_set)
        train_len = int(total_len * 0.9)
        val_len = total_len - train_len
        generator = torch.Generator().manual_seed(
            config.random_seed
        )  # fixed generator for reproducible splits to train models separately on same data splits
        train_subset, val_subset = torch.utils.data.random_split(train_set, [train_len, val_len], generator=generator)

        _save_split_indices({"train_indices": train_subset.indices, "val_indices": val_subset.indices}, config)

    indices = _load_split_indices(config)
    train_idx, val_idx = indices["train_indices"], indices["val_indices"]
    train_set = torch.utils.data.Subset(train_set, train_idx)
    val_set = torch.utils.data.Subset(train_set.dataset, val_idx)

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
