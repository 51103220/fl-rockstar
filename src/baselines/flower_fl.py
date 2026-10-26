"""
Flower (flwr) FL wrapper — FedAvg baseline for full Kyma reproduction.
Used when --algo fl in run_experiment.py on Kyma (multi-process setup).

For local quick tests, run_experiment.py uses the built-in run_fl() instead.
This wrapper is for real distributed Flower deployment across pods.

Usage (Kyma multi-pod):
  # Server pod
  python src/baselines/flower_fl.py --role server --n-rounds 1000

  # Client pods (one per node)
  python src/baselines/flower_fl.py --role client --node-id 0 \
      --server-address server-pod:8080 --task mnist
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

try:
    import flwr as fl
    from flwr.common import NDArrays, Scalar
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    print("[warn] flwr not installed — Flower baseline unavailable")


def get_params(model: nn.Module) -> NDArrays:
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_params(model: nn.Module, params: NDArrays) -> None:
    keys = list(model.state_dict().keys())
    sd = {k: torch.from_numpy(np.copy(v)) for k, v in zip(keys, params)}
    model.load_state_dict(sd, strict=True)


class HealFlowerClient(fl.client.NumPyClient if FLWR_AVAILABLE else object):
    """Flower client wrapping a HealClient-compatible node."""

    def __init__(self, node_id: int, model: nn.Module, train_loader, test_loader, device: str):
        self.node_id = node_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.device = device
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config) -> NDArrays:
        return get_params(self.model)

    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        set_params(self.model, parameters)
        self.model.train()
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X), y)
            loss.backward()
            self.optimizer.step()
        return get_params(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        set_params(self.model, parameters)
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += len(y)
        acc = correct / total if total > 0 else 0.0
        return float(1 - acc), total, {"accuracy": acc}


def run_server(n_rounds: int, n_clients: int, min_fit: int = None) -> None:
    if not FLWR_AVAILABLE:
        raise RuntimeError("Install flwr: pip install flwr")
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=min_fit or max(2, n_clients // 2),
        min_evaluate_clients=min_fit or max(2, n_clients // 2),
        min_available_clients=max(2, n_clients // 2),
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
    )


def run_client(node_id: int, server_address: str, task: str, data_dir: str, device: str) -> None:
    if not FLWR_AVAILABLE:
        raise RuntimeError("Install flwr: pip install flwr")
    from src.model.lenet5 import LeNet5, LogisticRegression
    from src.data.mnist_loader import make_noniid_mnist
    from src.data.spambase_loader import make_noniid_spambase

    if task == "mnist":
        node_loaders, test_loader = make_noniid_mnist(data_root=data_dir)
        model = LeNet5()
    else:
        node_loaders, test_loader = make_noniid_spambase(data_root=data_dir)
        model = LogisticRegression()

    client = HealFlowerClient(node_id, model, node_loaders[node_id], test_loader, device)
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["server", "client"], required=True)
    p.add_argument("--node-id", type=int, default=0)
    p.add_argument("--server-address", default="localhost:8080")
    p.add_argument("--n-rounds", type=int, default=1000)
    p.add_argument("--n-clients", type=int, default=100)
    p.add_argument("--task", choices=["mnist", "spambase"], default="mnist")
    p.add_argument("--data-dir", default="./data")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.role == "server":
        run_server(args.n_rounds, args.n_clients)
    else:
        run_client(args.node_id, args.server_address, args.task, args.data_dir, device)
