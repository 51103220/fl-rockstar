import sys
sys.path.insert(0, '..')

import copy
import random
import torch
import pytest

from src.model import SimpleCNN
from src.data import make_noniid_cifar10
from src.topology import (
    build_elevator_topology,
    build_ring_topology,
    heal_recover_hubs,
)
from src.simulation import HealClient, HealHub, run_heal_cycle
from src.failures import inject_crash
from src.evaluation import evaluate_model, evaluate_all


# ── Model ────────────────────────────────────────────────────────────────────

def test_simplecnn_forward():
    model = SimpleCNN()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"


def test_simplecnn_param_count():
    model = SimpleCNN()
    n = sum(p.numel() for p in model.parameters())
    assert 400_000 < n < 2_000_000, f"Unexpected param count: {n}"


# ── Topology ─────────────────────────────────────────────────────────────────

def test_elevator_diameter():
    import networkx as nx
    G, hub_ids = build_elevator_topology(n_nodes=20, n_hubs=3, seed=0)
    assert nx.is_connected(G)
    assert nx.diameter(G) == 2, "Elevator topology must have diameter 2"


def test_ring_topology():
    import networkx as nx
    G, hub_ids = build_ring_topology(n_nodes=20)
    assert hub_ids == []
    assert nx.is_connected(G)
    assert all(d == 2 for _, d in G.degree())  # ring: every node degree 2


def test_hub_recovery():
    G, hub_ids = build_elevator_topology(n_nodes=20, n_hubs=3, seed=0)
    failed = hub_ids[:1]
    G2, new_hubs = heal_recover_hubs(G, list(hub_ids), failed, n_hubs=3)
    assert len(new_hubs) == 3
    assert failed[0] not in new_hubs


# ── Data ─────────────────────────────────────────────────────────────────────

def test_noniid_split_shape():
    node_loaders, test_loader = make_noniid_cifar10(
        n_nodes=10, shards_per_node=2, batch_size=32, seed=0
    )
    assert len(node_loaders) == 10
    for loader in node_loaders.values():
        assert len(loader) > 0


def test_noniid_nonuniform():
    """Each node should have data from at most 2 classes."""
    node_loaders, _ = make_noniid_cifar10(
        n_nodes=10, shards_per_node=2, batch_size=256, seed=0
    )
    for node_id, loader in node_loaders.items():
        classes_seen = set()
        for _, y in loader:
            classes_seen.update(y.tolist())
        assert len(classes_seen) <= 2, (
            f"Node {node_id} has {len(classes_seen)} classes, expected ≤ 2"
        )


# ── HEAL Simulation ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_setup():
    """10 nodes, 2 hubs, 2 training cycles — fast smoke test."""
    n_nodes = 10
    n_hubs  = 2
    seed    = 0
    node_loaders, test_loader = make_noniid_cifar10(
        n_nodes=n_nodes, shards_per_node=2, batch_size=32, seed=seed
    )
    init_model = SimpleCNN()
    G, hub_ids = build_elevator_topology(n_nodes, n_hubs, seed)
    hubs    = [HealHub(i,    copy.deepcopy(init_model), node_loaders[i]) for i in hub_ids]
    clients = [HealClient(i, copy.deepcopy(init_model), node_loaders[i])
               for i in range(n_nodes) if i not in hub_ids]
    return clients, hubs, hub_ids, G, test_loader


def test_heal_cycle_runs(small_setup):
    clients, hubs, hub_ids, G, test_loader = small_setup
    # Should complete without exception
    run_heal_cycle(clients, hubs, n_sends=1)


def test_evaluate_all_returns_float(small_setup):
    clients, hubs, _, _, test_loader = small_setup
    acc = evaluate_all(clients + hubs, test_loader)
    assert 0.0 <= acc <= 1.0


def test_inject_crash_reduces_nodes(small_setup):
    clients, hubs, hub_ids, G, _ = small_setup
    n_before = len(clients) + len(hubs)
    new_clients, new_hubs, _, _ = inject_crash(
        list(clients), list(hubs), list(hub_ids), G,
        crash_rate=0.3, n_hubs=2, seed=0
    )
    n_after = len(new_clients) + len(new_hubs)
    assert n_after < n_before
