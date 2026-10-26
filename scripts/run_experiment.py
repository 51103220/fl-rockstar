#!/usr/bin/env python3
"""
Main entry point for all 40 HEAL reproduction experiments.

Usage:
  # Full Kyma run (100 nodes, 1000 cycles)
  python scripts/run_experiment.py --exp e1 --algo heal --task mnist \
      --n-nodes 100 --n-cycles 1000

  # Quick local test (10 nodes, 30 cycles, 2 hubs)
  python scripts/run_experiment.py --exp e1 --algo heal --task mnist --quick

  # Custom local run
  python scripts/run_experiment.py --exp e1 --algo heal --task mnist \
      --n-nodes 10 --n-hubs 2 --n-cycles 30 --eval-every 5

Experiments:
  e1  Crash-free
  e2  20% random node crash at cycle 100
  e3  All 5 hubs crash at cycle 100 (hub re-election)
  e4  10% churn every 50 cycles

Algorithms:
  heal          Hub-Enhanced Adaptive Learning (paper)
  fl            Federated Learning (FedAvg, star topology)
  gossip        Gossip Learning (random pairwise)
  epidemic      Epidemic Learning (push-pull gossip)
  epidemic-ring Epidemic Learning on ring topology
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import networkx as nx

from src.simulation.heal_node import HealClient, HealHub
from src.simulation.runner import run_simulation
from src.topology.elevator import build_elevator_topology


# ─────────────────────────── argument parsing ────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp",    choices=["e1","e2","e3","e4"], required=True)
    p.add_argument("--algo",   choices=["heal","fl","gossip","epidemic","epidemic-ring"], required=True)
    p.add_argument("--task",   choices=["mnist","spambase"], required=True)
    p.add_argument("--n-nodes",  type=int, default=100)
    p.add_argument("--n-hubs",   type=int, default=5)
    p.add_argument("--n-cycles", type=int, default=1000)
    p.add_argument("--n-sends",  type=int, default=2,  help="Hubs each client contacts")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--shards",   type=int, default=2,  help="Shards per node (non-IID level)")
    p.add_argument("--lr",       type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--data-dir",    default="./data")
    p.add_argument("--ckpt-dir",    default="./checkpoints")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 10 nodes, 2 hubs, 30 cycles, eval every 5")
    return p.parse_args()


# ─────────────────────────── data loading ────────────────────────────────────

def load_data(task: str, n_nodes: int, shards: int, batch_size: int, seed: int, data_dir: str):
    if task == "mnist":
        from src.data.mnist_loader import make_noniid_mnist
        return make_noniid_mnist(
            n_nodes=n_nodes, shards_per_node=shards,
            batch_size=batch_size, seed=seed, data_root=data_dir,
        )
    elif task == "spambase":
        from src.data.spambase_loader import make_noniid_spambase
        return make_noniid_spambase(
            n_nodes=n_nodes, shards_per_node=shards,
            batch_size=batch_size, seed=seed, data_root=data_dir,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# ─────────────────────────── model factory ───────────────────────────────────

def make_model(task: str):
    if task == "mnist":
        from src.model.lenet5 import LeNet5
        return LeNet5(num_classes=10)
    elif task == "spambase":
        from src.model.lenet5 import LogisticRegression
        return LogisticRegression(in_features=57, num_classes=2)
    else:
        raise ValueError(f"Unknown task: {task}")


# ─────────────────────────── topology / node setup ───────────────────────────

def build_nodes(n_nodes, n_hubs, node_loaders, task, lr, device):
    model_fn = lambda: make_model(task)

    # First n_hubs node IDs are hubs
    hub_ids = list(range(n_hubs))
    hubs = [
        HealHub(i, model_fn(), node_loaders[i], lr=lr, device=device)
        for i in hub_ids
    ]
    clients = [
        HealClient(i, model_fn(), node_loaders[i], lr=lr, device=device)
        for i in range(n_hubs, n_nodes)
    ]
    return clients, hubs, hub_ids


def build_ring(n_nodes):
    return nx.cycle_graph(n_nodes)


# ─────────────────────────── failure schedules ───────────────────────────────

def make_failure_schedule(exp: str, n_nodes: int, n_hubs: int, seed: int):
    """Return dict: {cycle: fn(clients, hubs, hub_ids, G) -> (clients, hubs, hub_ids, G)}"""
    rng = random.Random(seed)

    if exp == "e1":
        return {}

    if exp == "e2":
        # 20% random crash at cycle 100
        def crash_20pct(clients, hubs, hub_ids, G):
            all_c = clients + hubs
            n_crash = max(1, int(0.2 * len(all_c)))
            to_crash = rng.sample(all_c, n_crash)
            crash_ids = {n.id for n in to_crash}
            new_clients = [c for c in clients if c.id not in crash_ids]
            new_hubs    = [h for h in hubs    if h.id not in crash_ids]
            new_hub_ids = [i for i in hub_ids  if i not in crash_ids]
            new_G       = G.copy()
            for nid in crash_ids:
                if nid in new_G:
                    new_G.remove_node(nid)
            return new_clients, new_hubs, new_hub_ids, new_G
        return {100: crash_20pct}

    if exp == "e3":
        # All hubs crash at cycle 100, trigger re-election
        def crash_all_hubs(clients, hubs, hub_ids, G):
            # Elect new hubs from surviving clients (lowest-id first)
            n_new_hubs = len(hubs)
            new_hubs = []
            remaining_clients = list(clients)
            for c in remaining_clients[:n_new_hubs]:
                hub = HealHub(
                    c.id, c.model, c.data_loader,
                    lr=c.optimizer.param_groups[0]["lr"],
                    device=c.device,
                )
                new_hubs.append(hub)
            new_clients = remaining_clients[n_new_hubs:]
            new_hub_ids = [h.id for h in new_hubs]
            return new_clients, new_hubs, new_hub_ids, G
        return {100: crash_all_hubs}

    if exp == "e4":
        # 10% churn every 50 cycles: crash + rejoin
        schedule = {}
        def make_churn(cycle_seed):
            r = random.Random(cycle_seed)
            def churn(clients, hubs, hub_ids, G):
                all_c = clients + hubs
                n_churn = max(1, int(0.1 * len(all_c)))
                to_leave = r.sample(all_c, n_churn)
                leave_ids = {n.id for n in to_leave}
                new_clients = [c for c in clients if c.id not in leave_ids]
                new_hubs    = [h for h in hubs    if h.id not in leave_ids]
                new_hub_ids = [i for i in hub_ids  if i not in leave_ids]
                return new_clients, new_hubs, new_hub_ids, G
            return churn
        for c in range(50, 1001, 50):
            schedule[c] = make_churn(seed + c)
        return schedule

    raise ValueError(f"Unknown exp: {exp}")


# ─────────────────────────── FL (FedAvg) baseline ────────────────────────────

def run_fl(
    n_cycles: int,
    clients: list,
    test_loader,
    eval_every: int,
    ckpt_dir: str,
    local_epochs: int = 1,
):
    """Centralised FedAvg — server averages all clients every round."""
    from src.evaluation.metrics import evaluate_model

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "fl", "ckpt.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    start_cycle = 0
    history = []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for c in clients:
            if c.id in ckpt["node_states"]:
                c.model.load_state_dict(ckpt["node_states"][c.id])
        start_cycle = ckpt["cycle"] + 1
        history = ckpt["history"]
        print(f"  [resume] FL from cycle {start_cycle}")

    t0 = time.time()
    for cycle in range(start_cycle, n_cycles):
        # All clients train locally
        states = [c.train_local(epochs=local_epochs) for c in clients]
        # Server aggregates (equal weights)
        avg_sd = {}
        for key in states[0]:
            avg_sd[key] = torch.stack([s[key].float() for s in states]).mean(0)
        # Broadcast back
        for c in clients:
            c.model.load_state_dict(avg_sd)

        if cycle % eval_every == 0:
            device = getattr(clients[0], "device", "cpu")
            accs = [evaluate_model(c.model, test_loader, device) for c in clients[:10]]
            acc = sum(accs) / len(accs)
            elapsed = time.time() - t0
            cycles_done = cycle - start_cycle + 1
            eta = (elapsed / cycles_done) * (n_cycles - cycle - 1)
            print(f"  cycle={cycle:4d} | fl     | acc={acc:.4f} | "
                  f"alive={len(clients)} | eta={eta/60:.1f}m")
            history.append((cycle, acc))
            torch.save({
                "cycle": cycle, "history": history,
                "node_states": {c.id: c.get_state() for c in clients},
            }, ckpt_path)

    return history


# ─────────────────────────── Epidemic Learning ───────────────────────────────

def run_epidemic_cycle(clients: list, local_epochs: int = 1, ring: nx.Graph = None) -> None:
    """Push-pull epidemic: each node exchanges with a random peer (or ring neighbor)."""
    local_models = {c.id: c.train_local(epochs=local_epochs) for c in clients}
    alive = list(clients)

    for c in alive:
        if ring is not None:
            neighbors = [x for x in alive if x.id in ring.neighbors(c.id)]
        else:
            neighbors = [x for x in alive if x.id != c.id]

        if neighbors:
            partner = random.choice(neighbors)
            avg_sd = {}
            for key in local_models[c.id]:
                avg_sd[key] = torch.stack([
                    local_models[c.id][key].float(),
                    local_models[partner.id][key].float(),
                ]).mean(0)
            c.model.load_state_dict(avg_sd)
            partner.model.load_state_dict(avg_sd)


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    args = parse_args()

    if args.quick:
        args.n_nodes   = min(args.n_nodes, 10)
        args.n_hubs    = min(args.n_hubs, 2)
        args.n_cycles  = min(args.n_cycles, 30)
        args.eval_every = min(args.eval_every, 5)
        print(f"[quick] n_nodes={args.n_nodes} n_hubs={args.n_hubs} "
              f"n_cycles={args.n_cycles} eval_every={args.eval_every}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  exp={args.exp}  algo={args.algo}  task={args.task}")

    # ── data ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    node_loaders, test_loader = load_data(
        args.task, args.n_nodes, args.shards,
        args.batch_size, args.seed, args.data_dir,
    )

    # ── nodes ─────────────────────────────────────────────────────────────────
    print("Building nodes...")
    clients, hubs, hub_ids = build_nodes(
        args.n_nodes, args.n_hubs, node_loaders, args.task, args.lr, device
    )

    # ── topology ──────────────────────────────────────────────────────────────
    G, _ = build_elevator_topology(args.n_nodes, args.n_hubs, seed=args.seed)
    ring = build_ring(args.n_nodes)

    # ── failure schedule ──────────────────────────────────────────────────────
    failure_schedule = make_failure_schedule(args.exp, args.n_nodes, args.n_hubs, args.seed)

    # ── run ───────────────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.ckpt_dir, args.exp, args.algo, args.task)
    results_dir = os.path.join(args.results_dir, args.exp, args.algo, args.task)
    os.makedirs(results_dir, exist_ok=True)

    t0 = time.time()

    if args.algo == "heal":
        history = run_simulation(
            condition="heal",
            n_cycles=args.n_cycles,
            clients=clients,
            hubs=hubs,
            hub_ids=hub_ids,
            G=G,
            test_loader=test_loader,
            failure_schedule=failure_schedule,
            eval_every=args.eval_every,
            n_sends=args.n_sends,
            ckpt_dir=ckpt_dir,
            local_epochs=args.local_epochs,
        )

    elif args.algo == "fl":
        all_clients = clients + hubs  # treat all as FL clients
        history = run_fl(
            n_cycles=args.n_cycles,
            clients=all_clients,
            test_loader=test_loader,
            eval_every=args.eval_every,
            ckpt_dir=ckpt_dir,
            local_epochs=args.local_epochs,
        )

    elif args.algo == "gossip":
        all_clients = clients + hubs
        history = run_simulation(
            condition="gossip",
            n_cycles=args.n_cycles,
            clients=all_clients,
            hubs=[],
            hub_ids=[],
            G=G,
            test_loader=test_loader,
            failure_schedule=failure_schedule,
            eval_every=args.eval_every,
            ckpt_dir=ckpt_dir,
            local_epochs=args.local_epochs,
        )

    elif args.algo == "epidemic":
        all_clients = clients + hubs
        history = _run_generic(
            "epidemic", args.n_cycles, all_clients, test_loader,
            args.eval_every, ckpt_dir, args.local_epochs,
            failure_schedule=failure_schedule, ring=None,
        )

    elif args.algo == "epidemic-ring":
        all_clients = clients + hubs
        history = _run_generic(
            "epidemic-ring", args.n_cycles, all_clients, test_loader,
            args.eval_every, ckpt_dir, args.local_epochs,
            failure_schedule=failure_schedule, ring=ring,
        )

    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    total = time.time() - t0
    print(f"\nDone in {total/60:.1f}m. Cycles: {len(history)} eval points.")

    # ── save results ──────────────────────────────────────────────────────────
    result = {
        "exp": args.exp, "algo": args.algo, "task": args.task,
        "n_nodes": args.n_nodes, "n_hubs": args.n_hubs,
        "n_cycles": args.n_cycles, "seed": args.seed,
        "history": history,
        "final_acc": history[-1][1] if history else 0.0,
        "runtime_min": total / 60,
    }
    out_path = os.path.join(results_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results → {out_path}")
    print(f"Final accuracy: {result['final_acc']:.4f}")


def _run_generic(
    condition: str,
    n_cycles: int,
    clients: list,
    test_loader,
    eval_every: int,
    ckpt_dir: str,
    local_epochs: int,
    failure_schedule: dict,
    ring,
):
    """Generic loop for epidemic and epidemic-ring (not in runner.py)."""
    from src.evaluation.metrics import evaluate_all

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")

    start_cycle = 0
    history = []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for c in clients:
            if c.id in ckpt["node_states"]:
                c.model.load_state_dict(ckpt["node_states"][c.id])
        start_cycle = ckpt["cycle"] + 1
        history = ckpt["history"]
        print(f"  [resume] {condition} from cycle {start_cycle}")

    t0 = time.time()
    for cycle in range(start_cycle, n_cycles):
        if failure_schedule and cycle in failure_schedule:
            dummy_hubs = []
            dummy_hub_ids = []
            import networkx as nx_local
            clients, dummy_hubs, dummy_hub_ids, _ = failure_schedule[cycle](
                clients, dummy_hubs, dummy_hub_ids, nx_local.Graph()
            )

        run_epidemic_cycle(clients, local_epochs=local_epochs, ring=ring)

        if cycle % eval_every == 0:
            acc = evaluate_all(clients, test_loader)
            elapsed = time.time() - t0
            cycles_done = cycle - start_cycle + 1
            eta = (elapsed / cycles_done) * (n_cycles - cycle - 1)
            print(f"  cycle={cycle:4d} | {condition:14s} | acc={acc:.4f} | "
                  f"alive={len(clients)} | eta={eta/60:.1f}m")
            history.append((cycle, acc))
            torch.save({
                "cycle": cycle, "history": history,
                "node_states": {c.id: c.get_state() for c in clients},
            }, ckpt_path)

    return history


if __name__ == "__main__":
    main()
