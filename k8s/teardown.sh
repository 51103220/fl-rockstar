#!/usr/bin/env bash
# Teardown: delete all jobs and optionally PVCs
# Usage: bash k8s/teardown.sh [--keep-data]
NS=heal-repro
KEEP_DATA=${1:-""}

echo "=== Tearing down HEAL reproduction ==="

kubectl delete jobs --all -n $NS 2>/dev/null || true
kubectl delete pod results-collector -n $NS 2>/dev/null || true

if [[ "$KEEP_DATA" != "--keep-data" ]]; then
  echo "Deleting PVCs (use --keep-data to preserve checkpoints/results)..."
  kubectl delete -f k8s/pvc.yaml 2>/dev/null || true
else
  echo "Keeping PVCs (checkpoints + results preserved)."
fi

echo "Done."
