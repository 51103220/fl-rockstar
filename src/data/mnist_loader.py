import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset


def _num_workers() -> int:
    return 0 if os.uname().sysname == "Darwin" else 2


def make_noniid_mnist(
    n_nodes: int = 100,
    shards_per_node: int = 2,
    batch_size: int = 32,
    seed: int = 42,
    data_root: str = "./data",
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    """
    Shard-based non-IID MNIST split.

    shards_per_node=2 → highly heterogeneous (2 digits per node)
    shards_per_node=5 → moderate heterogeneity
    """
    random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = MNIST(root=data_root, train=True,  download=True, transform=transform)
    testset  = MNIST(root=data_root, train=False, download=True, transform=transform)

    labels = np.array(trainset.targets)
    sorted_idx = np.argsort(labels).tolist()

    n_shards = n_nodes * shards_per_node
    shard_size = len(trainset) // n_shards
    shards = [sorted_idx[i * shard_size:(i + 1) * shard_size] for i in range(n_shards)]
    random.shuffle(shards)

    node_loaders: Dict[int, DataLoader] = {}
    for i in range(n_nodes):
        idx: list = []
        for s in shards[i * shards_per_node:(i + 1) * shards_per_node]:
            idx.extend(s)
        subset = Subset(trainset, idx)
        node_loaders[i] = DataLoader(
            subset, batch_size=batch_size, shuffle=True,
            num_workers=_num_workers(), pin_memory=False,
        )

    test_loader = DataLoader(
        testset, batch_size=256, shuffle=False,
        num_workers=_num_workers(), pin_memory=False,
    )
    return node_loaders, test_loader
