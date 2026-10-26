import copy
import json
import random
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch

from src.data import make_noniid_cifar10
from src.evaluation import compute_recovery_metrics
from src.failures import inject_crash, inject_churn
from src.model import SimpleCNN
from src.simulation import HealClient, HealHub, run_simulation
from src.topology import (
    build_elevator_topology,
    build_gossip_topology,
    build_ring_topology,
)

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEEDS        = [42, 123, 256, 512, 999]
N_NODES      = 100
N_HUBS       = 5
N_CYCLES_E1  = 1000   # crash-free baseline
N_CYCLES_E2  = 1000   # 20% crash @ cycle 10
N_CYCLES_E3  = 500    # all-hubs fail @ cycle 100
N_CYCLES_E4  = 500    # sustained 10% churn every 50 cycles
EVAL_EVERY   = 10
RESULTS_DIR  = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)


def _build_nodes(
    node_data: Dict,
    init_model: torch.nn.Module,
    hub_ids: List[int],
    condition: str,
) -> tuple:
    if condition == "heal":
        hubs    = [HealHub(i,    copy.deepcopy(init_model), node_data[i]) for i in hub_ids]
        clients = [HealClient(i, copy.deepcopy(init_model), node_data[i])
                   for i in range(N_NODES) if i not in hub_ids]
    else:
        hubs    = []
        clients = [HealClient(i, copy.deepcopy(init_model), node_data[i])
                   for i in range(N_NODES)]
    return clients, hubs


def run_all(seed: int) -> Dict:
    random.seed(seed)
    torch.manual_seed(seed)

    node_data, test_loader = make_noniid_cifar10(N_NODES, shards_per_node=2, seed=seed)
    init_model = SimpleCNN()
    results: Dict = {}

    # ── E1: Crash-Free ──────────────────────────────────────────────────────
    print(f"\n=== E1 Crash-Free (seed={seed}) ===")
    for cond in ["heal", "ring", "gossip"]:
        G, hub_ids = (
            build_elevator_topology(N_NODES, N_HUBS, seed)
            if cond == "heal"
            else (build_ring_topology(N_NODES)
                  if cond == "ring"
                  else build_gossip_topology(N_NODES))
        )
        clients, hubs = _build_nodes(node_data, init_model, hub_ids, cond)
        hist = run_simulation(cond, N_CYCLES_E1, clients, hubs,
                               hub_ids, G, test_loader, eval_every=EVAL_EVERY)
        results[f"E1_{cond}"] = hist

    # ── E2: 20% Crash at Cycle 10 ───────────────────────────────────────────
    print(f"\n=== E2 20% Crash @ Cycle 10 (seed={seed}) ===")
    for cond in ["heal", "ring", "gossip"]:
        G, hub_ids = (
            build_elevator_topology(N_NODES, N_HUBS, seed)
            if cond == "heal"
            else (build_ring_topology(N_NODES)
                  if cond == "ring"
                  else build_gossip_topology(N_NODES))
        )
        clients, hubs = _build_nodes(node_data, init_model, hub_ids, cond)
        failure_schedule = {
            10: partial(inject_crash, crash_rate=0.2, n_hubs=N_HUBS, seed=seed)
        }
        hist = run_simulation(cond, N_CYCLES_E2, clients, hubs,
                               hub_ids, G, test_loader,
                               failure_schedule=failure_schedule,
                               eval_every=EVAL_EVERY)
        results[f"E2_{cond}"] = hist
        if cond == "heal":
            results["E2_heal_recovery"] = compute_recovery_metrics(hist, 10)

    # ── E3: All Hubs Fail at Cycle 100 ──────────────────────────────────────
    print(f"\n=== E3 All Hubs Fail @ Cycle 100 (seed={seed}) ===")
    G, hub_ids = build_elevator_topology(N_NODES, N_HUBS, seed)
    clients, hubs = _build_nodes(node_data, init_model, hub_ids, "heal")

    def _kill_all_hubs(c, h, hi, g):
        from src.failures import inject_crash
        return inject_crash(c, h, hi, g, crash_rate=len(h) / max(1, len(c + h)),
                            n_hubs=N_HUBS, seed=seed)

    failure_schedule = {100: _kill_all_hubs}
    hist = run_simulation("heal", N_CYCLES_E3, clients, hubs,
                           hub_ids, G, test_loader,
                           failure_schedule=failure_schedule,
                           eval_every=EVAL_EVERY)
    results["E3_heal"] = hist
    results["E3_heal_recovery"] = compute_recovery_metrics(hist, 100)

    # ── E4: Sustained 10% Churn Every 50 Cycles ─────────────────────────────
    print(f"\n=== E4 Sustained Churn (seed={seed}) ===")
    for cond in ["heal", "ring", "gossip"]:
        G, hub_ids = (
            build_elevator_topology(N_NODES, N_HUBS, seed)
            if cond == "heal"
            else (build_ring_topology(N_NODES)
                  if cond == "ring"
                  else build_gossip_topology(N_NODES))
        )
        clients, hubs = _build_nodes(node_data, init_model, hub_ids, cond)
        churn_sched = {
            cyc: partial(inject_churn, node_data=node_data,
                         init_model=init_model, churn_rate=0.10)
            for cyc in range(0, N_CYCLES_E4, 50)
        }
        hist = run_simulation(cond, N_CYCLES_E4, clients, hubs,
                               hub_ids, G, test_loader,
                               failure_schedule=churn_sched,
                               eval_every=EVAL_EVERY)
        results[f"E4_{cond}"] = hist

    return results


if __name__ == "__main__":
    all_results: Dict = {}
    for seed in SEEDS:
        seed_results = run_all(seed)
        all_results[f"seed_{seed}"] = seed_results

    output_path = RESULTS_DIR / "raw_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
