import os
import random
import time

import torch
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

from src.simulation.heal_node import HealClient, HealHub
from src.evaluation.metrics import evaluate_all


FailureFn = Callable[
    [List[HealClient], List[HealHub], List[int], nx.Graph],
    Tuple[List[HealClient], List[HealHub], List[int], nx.Graph],
]


def run_heal_cycle(
    clients: List[HealClient],
    hubs: List[HealHub],
    n_sends: int = 2,
    local_epochs: int = 3,
) -> None:
    """One complete HEAL training cycle (Algorithms 1 + 2)."""
    # Step 1 — local training
    client_models = {c.id: c.train_local(epochs=local_epochs) for c in clients}

    # Step 2 — each client pushes to random subset of hubs
    for c in clients:
        chosen = random.sample(hubs, k=min(n_sends, len(hubs)))
        for hub in chosen:
            hub.receive_from_client(client_models[c.id])

    # Step 3 — each hub aggregates its client submissions
    hub_local = [hub.aggregate_clients() for hub in hubs]

    # Step 4 — hub-to-hub exchange
    for i, hub in enumerate(hubs):
        other = [m for j, m in enumerate(hub_local) if j != i]
        hub.aggregate_hubs(other)

    # Step 5 — hubs broadcast to clients
    for c in clients:
        chosen = random.sample(hubs, k=min(n_sends, len(hubs)))
        c.receive_global([hub.get_state() for hub in chosen])


def run_ring_cycle(
    clients: List[HealClient],
    ring_graph: nx.Graph,
    local_epochs: int = 3,
) -> None:
    """One cycle of static-ring DFL — Condition A baseline."""
    local_models = {c.id: c.train_local(epochs=local_epochs) for c in clients}
    alive_ids = set(local_models.keys())

    for c in clients:
        neighbors = [
            local_models[n]
            for n in ring_graph.neighbors(c.id)
            if n in alive_ids
        ]
        if neighbors:
            c.receive_global(neighbors)
        # isolated nodes: model unchanged (worst case for convergence)


def run_gossip_cycle(clients: List[HealClient], local_epochs: int = 3) -> None:
    """One cycle of gossip learning — Condition B baseline."""
    local_models = {c.id: c.train_local(epochs=local_epochs) for c in clients}
    alive = list(clients)

    for c in alive:
        others = [x for x in alive if x.id != c.id]
        if others:
            partner = random.choice(others)
            c.receive_global([local_models[partner.id]])


def _ckpt_path(condition: str, ckpt_dir: str) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"ckpt_{condition}.pt")


def _save_ckpt(path: str, cycle: int, history: list, nodes: list) -> None:
    torch.save({
        "cycle": cycle,
        "history": history,
        "node_states": {n.id: n.get_state() for n in nodes},
    }, path)


def _load_ckpt(path: str, nodes: list, device: str) -> Tuple[int, list]:
    ckpt = torch.load(path, map_location=device)
    for n in nodes:
        if n.id in ckpt["node_states"]:
            n.model.load_state_dict(ckpt["node_states"][n.id])
    print(f"  [resume] loaded checkpoint: cycle={ckpt['cycle']}, "
          f"last_acc={ckpt['history'][-1][1]:.4f}")
    return ckpt["cycle"], ckpt["history"]


def run_simulation(
    condition: str,
    n_cycles: int,
    clients: List[HealClient],
    hubs: List[HealHub],
    hub_ids: List[int],
    G: nx.Graph,
    test_loader,
    failure_schedule: Optional[Dict[int, FailureFn]] = None,
    eval_every: int = 10,
    n_sends: int = 2,
    ckpt_dir: str = "checkpoints",
) -> List[Tuple[int, float]]:
    """
    condition: 'heal' | 'ring' | 'gossip'
    Saves a checkpoint after every eval. Resumes automatically if one exists.
    Returns list of (cycle, avg_accuracy).
    """
    all_nodes = (clients + hubs) if condition == "heal" else clients
    ckpt = _ckpt_path(condition, ckpt_dir)
    start_cycle = 0
    history: List[Tuple[int, float]] = []

    if os.path.exists(ckpt):
        start_cycle, history = _load_ckpt(ckpt, all_nodes, getattr(all_nodes[0], "device", "cpu"))
        start_cycle += 1  # resume from next cycle

    t_start = time.time()

    for cycle in range(start_cycle, n_cycles):
        t_cycle = time.time()

        if failure_schedule and cycle in failure_schedule:
            clients, hubs, hub_ids, G = failure_schedule[cycle](
                clients, hubs, hub_ids, G
            )
            all_nodes = (clients + hubs) if condition == "heal" else clients

        if condition == "heal":
            if not hubs:
                prev_acc = history[-1][1] if history else 0.0
                history.append((cycle, prev_acc))
                continue
            run_heal_cycle(clients, hubs, n_sends=n_sends, local_epochs=3)
        elif condition == "ring":
            run_ring_cycle(clients, G, local_epochs=3)
        elif condition == "gossip":
            run_gossip_cycle(clients, local_epochs=3)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        if cycle % eval_every == 0:
            t_eval = time.time()
            acc = evaluate_all(all_nodes, test_loader)
            elapsed = time.time() - t_start
            cycles_done = cycle - start_cycle + 1
            eta = (elapsed / cycles_done) * (n_cycles - cycle - 1)
            print(f"  cycle={cycle:4d} | {condition:6s} | acc={acc:.4f} | "
                  f"alive={len(all_nodes)} | "
                  f"cycle={time.time()-t_cycle:.1f}s | eval={time.time()-t_eval:.1f}s | "
                  f"eta={eta/60:.1f}m")
            history.append((cycle, acc))
            _save_ckpt(ckpt, cycle, history, all_nodes)

    return history
