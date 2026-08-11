"""Small CIFAR-10 data helpers for raw-pixel UAP training and evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def cifar10_dataset(root: str, *, train: bool) -> Dataset:
    """Return CIFAR-10 tensors in [0, 1]; model normalization is done later."""

    return datasets.CIFAR10(root=root, train=train, transform=transforms.ToTensor(), download=False)


def loader(dataset: Dataset, *, batch_size: int, train: bool, workers: int) -> DataLoader:
    """Build a deterministic evaluation loader or a shuffled training loader."""

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=train,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
