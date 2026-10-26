import random
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
) -> None:
    """One complete HEAL training cycle (Algorithms 1 + 2)."""
    # Step 1 — local training
    client_models = {c.id: c.train_local(epochs=1) for c in clients}

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
) -> None:
    """One cycle of static-ring DFL — Condition A baseline."""
    local_models = {c.id: c.train_local(epochs=1) for c in clients}
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


def run_gossip_cycle(clients: List[HealClient]) -> None:
    """One cycle of gossip learning — Condition B baseline."""
    local_models = {c.id: c.train_local(epochs=1) for c in clients}
    alive = list(clients)

    for c in alive:
        others = [x for x in alive if x.id != c.id]
        if others:
            partner = random.choice(others)
            c.receive_global([local_models[partner.id]])


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
) -> List[Tuple[int, float]]:
    """
    condition: 'heal' | 'ring' | 'gossip'
    failure_schedule: {cycle_number: failure_fn}

    Returns list of (cycle, avg_accuracy).
    """
    history: List[Tuple[int, float]] = []

    import time
    t_start = time.time()

    for cycle in range(n_cycles):
        t_cycle = time.time()

        if failure_schedule and cycle in failure_schedule:
            clients, hubs, hub_ids, G = failure_schedule[cycle](
                clients, hubs, hub_ids, G
            )

        if condition == "heal":
            if not hubs:
                prev_acc = history[-1][1] if history else 0.0
                history.append((cycle, prev_acc))
                continue
            run_heal_cycle(clients, hubs, n_sends=n_sends)
        elif condition == "ring":
            run_ring_cycle(clients, G)
        elif condition == "gossip":
            run_gossip_cycle(clients)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        if cycle % eval_every == 0:
            t_eval = time.time()
            all_nodes = (clients + hubs) if condition == "heal" else clients
            acc = evaluate_all(all_nodes, test_loader)
            elapsed = time.time() - t_start
            eta = (elapsed / (cycle + 1)) * (n_cycles - cycle - 1)
            print(f"  cycle={cycle:4d} | {condition:6s} | acc={acc:.4f} | "
                  f"alive={len(all_nodes)} | "
                  f"cycle={time.time()-t_cycle:.1f}s | eval={time.time()-t_eval:.1f}s | "
                  f"eta={eta/60:.1f}m")
            history.append((cycle, acc))

    return history
