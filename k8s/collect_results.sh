#!/usr/bin/env bash
# Collect results from PVC into local ./results/<RUN_ID>/
# Usage: bash k8s/collect_results.sh [RUN_ID]
#   RUN_ID defaults to today's date (YYYYMMDD).
set -euo pipefail

NS=keda
RUN_ID="${1:-$(date +%Y%m%d)}"
LOCAL_RESULTS="./results/${RUN_ID}"
mkdir -p "$LOCAL_RESULTS"

echo "=== Collecting results from Kyma PVC → ${LOCAL_RESULTS} ==="

kubectl run results-collector \
  --image=docker-hub.common.repositories.cloud.sap/library/busybox \
  --restart=Never \
  --namespace=$NS \
  --overrides='{
    "spec": {
      "imagePullSecrets": [{"name":"artifactory-pull-secret"},{"name":"dockerhub-pull-secret"}],
      "containers": [{
        "name": "results-collector",
        "image": "docker-hub.common.repositories.cloud.sap/library/busybox",
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
kubectl wait --for=condition=ready pod/results-collector -n $NS --timeout=120s

echo "Copying..."
kubectl cp $NS/results-collector:/results "$LOCAL_RESULTS/results"
kubectl cp $NS/results-collector:/checkpoints "$LOCAL_RESULTS/checkpoints"

echo "Cleaning up collector pod..."
kubectl delete pod results-collector -n $NS

echo ""
echo "=== Saved to ${LOCAL_RESULTS} ==="
find "$LOCAL_RESULTS/results" -name "results.json" | wc -l | xargs -I{} echo "  {} result files"
echo ""
echo "Plot with:"
echo "  .venv/bin/python scripts/plot_results.py \\"
echo "    --results-dir ${LOCAL_RESULTS}/results \\"
echo "    --out-dir ${LOCAL_RESULTS}/plots"
