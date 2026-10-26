#!/usr/bin/env python3
"""
Plot results from all 40 experiments.
Reproduces Figure 4 and Table 2 from the HEAL paper.

Usage:
  python scripts/plot_results.py --results-dir ./results/full_reproduction
  python scripts/plot_results.py --results-dir ./results --exp e1
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


ALGO_NAMES = {
    "heal":          "HEAL",
    "fl":            "Federated Learning",
    "gossip":        "Gossip Learning",
    "epidemic":      "Epidemic Learning",
    "epidemic-ring": "Epidemic on Ring",
}
ALGO_COLORS = {
    "heal":          "#2196F3",
    "fl":            "#4CAF50",
    "gossip":        "#FF9800",
    "epidemic":      "#9C27B0",
    "epidemic-ring": "#F44336",
}
ALGO_MARKERS = {
    "heal":          "o",
    "fl":            "s",
    "gossip":        "^",
    "epidemic":      "D",
    "epidemic-ring": "x",
}
EXP_TITLES = {
    "e1": "E1: Crash-Free",
    "e2": "E2: 20% Node Crash",
    "e3": "E3: All Hubs Crash",
    "e4": "E4: 10% Churn/Cycle",
}


def load_results(results_dir: str) -> pd.DataFrame:
    rows = []
    for json_path in Path(results_dir).rglob("results.json"):
        with open(json_path) as f:
            d = json.load(f)
        rows.append(d)
    if not rows:
        print(f"No results.json found in {results_dir}")
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_accuracy_curves(df: pd.DataFrame, exp: str, task: str, out_dir: str) -> None:
    subset = df[(df["exp"] == exp) & (df["task"] == task)]
    if subset.empty:
        print(f"  No data for exp={exp} task={task}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in subset.iterrows():
        algo = row["algo"]
        history = row["history"]
        if not history:
            continue
        cycles = [h[0] for h in history]
        accs   = [h[1] for h in history]
        ax.plot(
            cycles, accs,
            label=ALGO_NAMES.get(algo, algo),
            color=ALGO_COLORS.get(algo, "gray"),
            marker=ALGO_MARKERS.get(algo, "o"),
            markevery=max(1, len(cycles) // 10),
            linewidth=2,
        )

    ax.set_xlabel("Training Cycles")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{EXP_TITLES.get(exp, exp)} — {task.upper()}")
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"accuracy_{exp}_{task}.pdf")
    fig.savefig(fname, bbox_inches="tight")
    png_fname = fname.replace(".pdf", ".png")
    fig.savefig(png_fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def print_summary_table(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print("\n" + "="*70)
    print("FINAL ACCURACY SUMMARY TABLE")
    print("="*70)
    pivot = df.pivot_table(
        index=["exp", "task"],
        columns="algo",
        values="final_acc",
        aggfunc="mean",
    )
    print(pivot.to_string(float_format="{:.4f}".format))
    print("="*70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-dir", default=None, help="Output dir for plots (default: results-dir/plots)")
    p.add_argument("--exp", default=None, help="Filter to one experiment (e1/e2/e3/e4)")
    p.add_argument("--task", default=None, help="Filter to one task (mnist/spambase)")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")
    print(f"Loading results from {args.results_dir}...")
    df = load_results(args.results_dir)

    if df.empty:
        print("No results found. Run experiments first.")
        sys.exit(1)

    print(f"Found {len(df)} result files.")
    print_summary_table(df)

    exps  = [args.exp]  if args.exp  else ["e1","e2","e3","e4"]
    tasks = [args.task] if args.task else ["mnist","spambase"]

    print(f"\nGenerating plots → {out_dir}/")
    for exp in exps:
        for task in tasks:
            print(f"  Plotting {exp} / {task}...")
            plot_accuracy_curves(df, exp, task, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
