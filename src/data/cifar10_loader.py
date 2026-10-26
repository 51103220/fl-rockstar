import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def _num_workers() -> int:
    # macOS + notebooks: fork-based multiprocessing causes deadlocks → use 0
    # Linux (Docker): 2 workers are safe
    return 0 if os.uname().sysname == "Darwin" else 2


class _HFCifar10Dataset(Dataset):
    """Wraps a HuggingFace CIFAR-10 split as a torch Dataset."""

    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = sample["img"]  # PIL Image
        label = sample["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


def make_noniid_cifar10(
    n_nodes: int = 100,
    shards_per_node: int = 2,
    batch_size: int = 32,
    seed: int = 42,
    # data_root kept for backward compat but unused — HF handles caching
    data_root: str = None,
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    """
    Shard-based non-IID CIFAR-10 split using HuggingFace datasets.

    shards_per_node=2  → highly heterogeneous (2 classes per node)
    shards_per_node=5  → mildly heterogeneous (5 classes per node)

    Returns:
        node_loaders: {node_id: DataLoader}
        test_loader:  DataLoader for global evaluation
    """
    random.seed(seed)
    torch.manual_seed(seed)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    print("Loading CIFAR-10 from HuggingFace (uoft-cs/cifar10)...")
    hf = load_dataset("uoft-cs/cifar10")

    trainset = _HFCifar10Dataset(hf["train"], transform=transform_train)
    testset  = _HFCifar10Dataset(hf["test"],  transform=transform_test)

    # Batch column fetch — ~100x faster than per-sample loop
    labels = np.array(hf["train"]["label"])
    sorted_idx = np.argsort(labels).tolist()

    n_shards = n_nodes * shards_per_node
    shard_size = len(trainset) // n_shards
    shards = [
        sorted_idx[i * shard_size : (i + 1) * shard_size]
        for i in range(n_shards)
    ]
    random.shuffle(shards)

    node_loaders: Dict[int, DataLoader] = {}
    for i in range(n_nodes):
        idx: list = []
        for s in shards[i * shards_per_node : (i + 1) * shards_per_node]:
            idx.extend(s)
        subset = torch.utils.data.Subset(trainset, idx)
        node_loaders[i] = DataLoader(
            subset, batch_size=batch_size, shuffle=True,
            num_workers=_num_workers(), pin_memory=False,
        )

    test_loader = DataLoader(
        testset, batch_size=256, shuffle=False,
        num_workers=_num_workers(), pin_memory=False,
    )
    return node_loaders, test_loader
