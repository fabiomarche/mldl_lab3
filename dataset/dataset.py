
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, List

def build_transforms() -> Tuple[v2.Compose, v2.Compose]:
    """ Transforms for training and eval set"""

    training_transform = v2.Compose([ # simple data augmentation
        v2.ToImage(),
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomVerticalFlip(p = 0.5),
        v2.RandomRotation(degrees = 30),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]) #note that the normalization in the validation set must be equal to what applied in training. This should speed up the computation
    ])
    
    return training_transform, val_transform


def build_datasets(data_dir: str ="data"):
    """ Returns (train_ds, val_ds). If val_split == 0 it uses directly the test set as val set"""

    train_t, val_t = build_transforms()
    

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=train_t,
    )

    val_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=val_t,
    )
    
    return training_data, val_data

def build_loaders(train_ds, val_ds, batch_size: int = 64, num_workers: int = 2):
    """It returns (train_loader, val_loader)"""
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def get_data_info() -> dict:
    """Utisl to config the model"""
    
    return {
        "in_channels": 1,
        "num_classes": 10,
        "image_size": (28,28),
        "class_names": datasets.FashionMNIST.classes,
    }