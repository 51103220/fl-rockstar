import copy
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader


class HealClient:
    """HEAL Algorithm 2 — client node."""

    def __init__(
        self,
        node_id: int,
        model: nn.Module,
        data_loader: DataLoader,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        device: str = "cpu",
    ):
        self.id = node_id
        self.model = copy.deepcopy(model).to(device)
        self.data_loader = data_loader
        self.alive = True
        self.device = device
        self.optimizer = SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_local(self, epochs: int = 1) -> nn.Module:
        """Algorithm 2, Step 1: local training."""
        self.model.train()
        for _ in range(epochs):
            for X, y in self.data_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                self.optimizer.step()
        return copy.deepcopy(self.model)

    def receive_global(self, hub_models: List[nn.Module]) -> None:
        """Algorithm 2, Steps 4–5: average models received from hubs."""
        if not hub_models:
            return
        avg_sd = copy.deepcopy(hub_models[0].state_dict())
        for key in avg_sd:
            stacked = torch.stack(
                [m.state_dict()[key].float() for m in hub_models]
            )
            avg_sd[key] = stacked.mean(0)
        self.model.load_state_dict(avg_sd)


class HealHub(HealClient):
    """HEAL Algorithm 1 — hub node (aggregator)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client_models: List[nn.Module] = []

    def receive_from_client(self, model: nn.Module) -> None:
        """Algorithm 1, Step 1: buffer an incoming client model."""
        self._client_models.append(copy.deepcopy(model))

    def aggregate_clients(self) -> nn.Module:
        """Algorithm 1, Step 2: FedAvg over buffered client models."""
        if not self._client_models:
            return copy.deepcopy(self.model)
        avg_sd = copy.deepcopy(self._client_models[0].state_dict())
        for key in avg_sd:
            stacked = torch.stack(
                [m.state_dict()[key].float() for m in self._client_models]
            )
            avg_sd[key] = stacked.mean(0)
        self.model.load_state_dict(avg_sd)
        self._client_models.clear()
        return copy.deepcopy(self.model)

    def aggregate_hubs(self, other_hub_models: List[nn.Module]) -> None:
        """Algorithm 1, Step 5: average own model with peer hub models."""
        all_models = [self.model] + other_hub_models
        avg_sd = copy.deepcopy(self.model.state_dict())
        for key in avg_sd:
            stacked = torch.stack(
                [m.state_dict()[key].float() for m in all_models]
            )
            avg_sd[key] = stacked.mean(0)
        self.model.load_state_dict(avg_sd)
