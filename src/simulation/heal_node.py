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
        self.model = model.to(device)
        self.data_loader = data_loader
        self.alive = True
        self.device = device
        self.optimizer = SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def get_state(self) -> dict:
        """Return a {key: tensor} snapshot kept on the model's device."""
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def train_local(self, epochs: int = 1) -> dict:
        """Algorithm 2, Step 1: local training. Returns state dict (no deepcopy)."""
        self.model.train()
        for _ in range(epochs):
            for X, y in self.data_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                self.optimizer.step()
        return self.get_state()

    def receive_global(self, hub_states: List[dict]) -> None:
        """Algorithm 2, Steps 4–5: average state dicts received from hubs."""
        if not hub_states:
            return
        avg_sd = {}
        for key in hub_states[0]:
            avg_sd[key] = torch.stack([s[key].float() for s in hub_states]).mean(0)
        self.model.load_state_dict(avg_sd)


class HealHub(HealClient):
    """HEAL Algorithm 1 — hub node (aggregator)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client_models: List[dict] = []

    def receive_from_client(self, state: dict) -> None:
        """Algorithm 1, Step 1: buffer an incoming client state dict."""
        self._client_models.append(state)

    def aggregate_clients(self) -> dict:
        """Algorithm 1, Step 2: FedAvg over buffered client state dicts."""
        if not self._client_models:
            return self.get_state()
        avg_sd = {}
        for key in self._client_models[0]:
            avg_sd[key] = torch.stack([s[key].float() for s in self._client_models]).mean(0)
        self.model.load_state_dict(avg_sd)
        self._client_models.clear()
        return self.get_state()

    def aggregate_hubs(self, other_hub_states: List[dict]) -> None:
        """Algorithm 1, Step 5: average own model with peer hub state dicts."""
        all_states = [self.get_state()] + other_hub_states
        avg_sd = {}
        for key in all_states[0]:
            avg_sd[key] = torch.stack([s[key].float() for s in all_states]).mean(0)
        self.model.load_state_dict(avg_sd)
