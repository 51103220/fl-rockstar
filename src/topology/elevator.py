import random
from typing import List, Tuple

import networkx as nx


def build_elevator_topology(
    n_nodes: int = 100,
    n_hubs: int = 5,
    seed: int = 42,
) -> Tuple[nx.Graph, List[int]]:
    """
    Elevator overlay (diameter = 2):
    - h hub nodes connected to ALL other nodes
    - non-hub nodes have sparse random peer connections (degree ~3)
    """
    random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    hub_ids = random.sample(range(n_nodes), n_hubs)

    for hub in hub_ids:
        for node in range(n_nodes):
            if node != hub:
                G.add_edge(hub, node)

    non_hubs = [n for n in range(n_nodes) if n not in hub_ids]
    for node in non_hubs:
        peers = random.sample([x for x in non_hubs if x != node], k=min(3, len(non_hubs) - 1))
        for peer in peers:
            G.add_edge(node, peer)

    return G, hub_ids


def build_ring_topology(n_nodes: int = 100) -> Tuple[nx.Graph, List[int]]:
    """Static ring — Condition A baseline (no healing)."""
    return nx.cycle_graph(n_nodes), []


def build_gossip_topology(n_nodes: int = 100) -> Tuple[nx.Graph, List[int]]:
    """Complete graph used for gossip pairing — Condition B baseline."""
    return nx.complete_graph(n_nodes), []


def heal_recover_hubs(
    G: nx.Graph,
    hub_ids: List[int],
    failed_ids: List[int],
    n_hubs: int = 5,
) -> Tuple[nx.Graph, List[int]]:
    """
    Re-elect hubs after failures using Elevator self-election:
    highest-degree surviving nodes become new hubs.
    New hubs connect to ALL surviving nodes (restores diameter=2).
    """
    for f in list(failed_ids):
        if f in G:
            G.remove_node(f)
        if f in hub_ids:
            hub_ids.remove(f)

    while len(hub_ids) < n_hubs:
        candidates = [n for n in G.nodes() if n not in hub_ids]
        if not candidates:
            break
        new_hub = max(candidates, key=lambda n: G.degree(n))
        hub_ids.append(new_hub)
        for node in G.nodes():
            if node != new_hub:
                G.add_edge(new_hub, node)

    return G, hub_ids
