from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


def evaluate_all(nodes: list, test_loader: DataLoader, device: str = "cpu", sample: int = 10) -> float:
    """Average accuracy across a random sample of nodes (default 10) for speed."""
    import random
    subset = random.sample(nodes, min(sample, len(nodes)))
    accs = [evaluate_model(n.model, test_loader, getattr(n, "device", device)) for n in subset]
    return sum(accs) / len(accs) if accs else 0.0


def compute_recovery_metrics(
    history: List[Tuple[int, float]],
    failure_cycle: int,
) -> Dict:
    """
    Computes pre-failure accuracy, max drop, recovery cycle, and cycles-to-recover
    from an accuracy history list of (cycle, acc) tuples.
    """
    pre = next(
        (acc for cyc, acc in history if cyc == failure_cycle - 10), None
    )
    post = [(cyc, acc) for cyc, acc in history if cyc >= failure_cycle]

    if not pre or not post:
        return {}

    min_acc = min(acc for _, acc in post)
    max_drop = pre - min_acc
    recovery_cycle = next(
        (cyc for cyc, acc in post if acc >= pre * 0.98), None
    )

    return {
        "pre_failure_acc": round(pre, 4),
        "max_drop": round(max_drop, 4),
        "recovery_cycle": recovery_cycle,
        "cycles_to_recover": (
            recovery_cycle - failure_cycle if recovery_cycle else None
        ),
        "final_acc": round(post[-1][1], 4),
    }
