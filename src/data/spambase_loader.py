import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler


def _num_workers() -> int:
    return 0 if os.uname().sysname == "Darwin" else 2


def _load_spambase(data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Spambase CSV. Downloads from UCI if not present."""
    path = os.path.join(data_root, "spambase.data")
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(data_root, exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        print(f"Downloading Spambase from UCI...")
        urllib.request.urlretrieve(url, path)
    data = np.loadtxt(path, delimiter=",")
    X, y = data[:, :-1].astype(np.float32), data[:, -1].astype(np.int64)
    return X, y


def make_noniid_spambase(
    n_nodes: int = 100,
    shards_per_node: int = 2,
    batch_size: int = 32,
    seed: int = 42,
    data_root: str = "./data",
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    """
    Shard-based non-IID Spambase split (binary: spam=1, ham=0).
    2 classes only — shards alternate spam/ham shards.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    X, y = _load_spambase(data_root)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # 80/20 train/test split
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:int(0.8 * n)], idx[int(0.8 * n):]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    sorted_idx = np.argsort(y_train).tolist()

    n_shards = n_nodes * shards_per_node
    shard_size = len(X_train) // n_shards
    shards = [sorted_idx[i * shard_size:(i + 1) * shard_size] for i in range(n_shards)]
    random.shuffle(shards)

    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train)
    full_train = TensorDataset(X_t, y_t)

    node_loaders: Dict[int, DataLoader] = {}
    for i in range(n_nodes):
        idx_node: list = []
        for s in shards[i * shards_per_node:(i + 1) * shards_per_node]:
            idx_node.extend(s)
        subset = Subset(full_train, idx_node)
        node_loaders[i] = DataLoader(
            subset, batch_size=batch_size, shuffle=True,
            num_workers=_num_workers(), pin_memory=False,
        )

    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=_num_workers(), pin_memory=False,
    )
    return node_loaders, test_loader
