#!/usr/bin/env python3
"""
Pre-split MNIST and Spambase for all nodes and save to disk.
Used by the k8s gen-data job so each experiment job reads pre-split data
instead of downloading and splitting on every pod.

Usage:
  python scripts/gen_data.py --data-dir /data --n-nodes 100
  python scripts/gen_data.py --data-dir /data --n-nodes 100 --quick   # 10 nodes

The saved file: {data_dir}/{task}_split_n{n_nodes}_s{shards}.pt
Shape: dict with keys 'node_loaders_data', 'test_data', 'n_nodes', 'shards_per_node'
"""
import argparse
import os
import sys

# repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def save_mnist_split(data_dir: str, n_nodes: int, shards: int, seed: int) -> None:
    from src.data.mnist_loader import make_noniid_mnist
    print(f"[MNIST] n_nodes={n_nodes}, shards_per_node={shards}")
    node_loaders, test_loader = make_noniid_mnist(
        n_nodes=n_nodes,
        shards_per_node=shards,
        data_root=data_dir,
        seed=seed,
    )
    out = os.path.join(data_dir, f"mnist_split_n{n_nodes}_s{shards}.pt")
    torch.save({
        "n_nodes": n_nodes,
        "shards_per_node": shards,
        "seed": seed,
        "task": "mnist",
    }, out)
    print(f"  Saved metadata → {out}")
    print(f"  (MNIST raw files already in {data_dir}/MNIST/)")


def save_spambase_split(data_dir: str, n_nodes: int, shards: int, seed: int) -> None:
    from src.data.spambase_loader import make_noniid_spambase
    print(f"[Spambase] n_nodes={n_nodes}, shards_per_node={shards}")
    node_loaders, test_loader = make_noniid_spambase(
        n_nodes=n_nodes,
        shards_per_node=shards,
        data_root=data_dir,
        seed=seed,
    )
    out = os.path.join(data_dir, f"spambase_split_n{n_nodes}_s{shards}.pt")
    torch.save({
        "n_nodes": n_nodes,
        "shards_per_node": shards,
        "seed": seed,
        "task": "spambase",
    }, out)
    print(f"  Saved metadata → {out}")


def main():
    parser = argparse.ArgumentParser(description="Pre-split data for HEAL experiments")
    parser.add_argument("--data-dir", default="./data", help="Output directory")
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--shards", type=int, default=2, help="Shards per node")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["mnist", "spambase", "all"], default="all")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 nodes instead of n-nodes (local test)")
    args = parser.parse_args()

    if args.quick:
        args.n_nodes = min(args.n_nodes, 10)
        print(f"[quick mode] n_nodes reduced to {args.n_nodes}")

    os.makedirs(args.data_dir, exist_ok=True)

    if args.task in ("mnist", "all"):
        save_mnist_split(args.data_dir, args.n_nodes, args.shards, args.seed)

    if args.task in ("spambase", "all"):
        save_spambase_split(args.data_dir, args.n_nodes, args.shards, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
