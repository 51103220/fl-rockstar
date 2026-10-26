#!/usr/bin/env bash
# Collect results from PVC into local ./results/ after all jobs complete
# Usage: bash k8s/collect_results.sh
set -euo pipefail

NS=heal-repro
LOCAL_RESULTS=./results/full_reproduction
mkdir -p "$LOCAL_RESULTS"

echo "=== Collecting results from Kyma PVC ==="

# Spin up a temporary pod to copy from PVC
kubectl run results-collector \
  --image=alpine \
  --restart=Never \
  --namespace=$NS \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "results-collector",
        "image": "alpine",
        "command": ["sleep", "3600"],
        "volumeMounts": [
          {"name": "results", "mountPath": "/results"},
          {"name": "checkpoints", "mountPath": "/checkpoints"}
        ]
      }],
      "volumes": [
        {"name": "results", "persistentVolumeClaim": {"claimName": "heal-results-pvc"}},
        {"name": "checkpoints", "persistentVolumeClaim": {"claimName": "heal-checkpoints-pvc"}}
      ]
    }
  }' 2>/dev/null || true

echo "Waiting for collector pod..."
kubectl wait --for=condition=ready pod/results-collector -n $NS --timeout=60s

echo "Copying results..."
kubectl cp $NS/results-collector:/results "$LOCAL_RESULTS/results"
kubectl cp $NS/results-collector:/checkpoints "$LOCAL_RESULTS/checkpoints"

echo "Cleaning up collector pod..."
kubectl delete pod results-collector -n $NS

echo ""
echo "=== Results saved to $LOCAL_RESULTS ==="
ls -lh "$LOCAL_RESULTS/results/" 2>/dev/null || echo "No results yet."

echo ""
echo "Generate plots with:"
echo "  python3 scripts/plot_results.py --results-dir $LOCAL_RESULTS/results"
