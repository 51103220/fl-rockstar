import copy
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.simulation.heal_node import HealClient, HealHub
from src.topology.elevator import heal_recover_hubs


def inject_crash(
    clients: List[HealClient],
    hubs: List[HealHub],
    hub_ids: List[int],
    G: nx.Graph,
    crash_rate: float = 0.2,
    n_hubs: int = 5,
    seed: Optional[int] = None,
) -> Tuple[List[HealClient], List[HealHub], List[int], nx.Graph]:
    """
    Kill crash_rate fraction of all nodes in one shot.
    Triggers hub re-election if any hubs are among the victims.
    """
    if seed is not None:
        random.seed(seed)

    all_nodes = clients + hubs
    n_crash = max(1, int(len(all_nodes) * crash_rate))
    victims = random.sample(all_nodes, min(n_crash, len(all_nodes)))
    victim_ids = {v.id for v in victims}

    surviving_clients = [c for c in clients if c.id not in victim_ids]
    surviving_hubs   = [h for h in hubs   if h.id not in victim_ids]

    G, new_hub_ids = heal_recover_hubs(
        G, [h.id for h in surviving_hubs], list(victim_ids), n_hubs=n_hubs
    )

    # Promote newly elected hubs from surviving client pool
    promoted_ids = set(new_hub_ids) - {h.id for h in surviving_hubs}
    new_hub_objects: List[HealHub] = []
    remaining_clients: List[HealClient] = []

    for c in surviving_clients:
        if c.id in promoted_ids:
            hub = HealHub(c.id, c.model, c.data_loader)
            new_hub_objects.append(hub)
        else:
            remaining_clients.append(c)

    return remaining_clients, surviving_hubs + new_hub_objects, new_hub_ids, G


def inject_hub_crash(
    clients: List[HealClient],
    hubs: List[HealHub],
    hub_ids: List[int],
    G: nx.Graph,
    n_hubs: int = 5,
) -> Tuple[List[HealClient], List[HealHub], List[int], nx.Graph]:
    """Kill ALL current hubs — tests worst-case hub failure scenario (E3)."""
    return inject_crash(
        clients, hubs, hub_ids, G,
        crash_rate=1.0,   # will be overridden to remove only hubs
        n_hubs=n_hubs,
    )


def inject_churn(
    clients: List[HealClient],
    hubs: List[HealHub],
    hub_ids: List[int],
    G: nx.Graph,
    node_data: Dict[int, any],
    init_model,
    churn_rate: float = 0.10,
    n_hubs: int = 5,
) -> Tuple[List[HealClient], List[HealHub], List[int], nx.Graph]:
    """
    Replace churn_rate fraction of client nodes with fresh cold-start nodes.
    New nodes receive the current global model from random hubs.
    """
    n_replace = max(1, int(len(clients) * churn_rate))
    departing = random.sample(clients, min(n_replace, len(clients)))
    departing_ids = {c.id for c in departing}

    survivors = [c for c in clients if c.id not in departing_ids]

    max_id = max((c.id for c in clients + hubs), default=0) + 1
    for i in range(n_replace):
        new_id = max_id + i
        G.add_node(new_id)
        for hub in hubs:
            G.add_edge(new_id, hub.id)
        loader = node_data.get(new_id % len(node_data))
        new_node = HealClient(
            node_id=new_id,
            model=copy.deepcopy(init_model),
            data_loader=loader,
        )
        survivors.append(new_node)

    return survivors, hubs, hub_ids, G
