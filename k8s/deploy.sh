#!/usr/bin/env bash
# Deploy the full HEAL reproduction pipeline to Kyma
# Usage: bash k8s/deploy.sh [e1|e2|e3|e4|all]
set -euo pipefail

EXP=${1:-all}
NS=heal-repro

echo "=== HEAL Reproduction Deploy ==="
echo "Experiment : $EXP"
echo "Namespace  : $NS"
echo ""

# 1. Namespace + storage
echo "[1/4] Applying namespace and PVCs..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/heal-checkpoints-pvc pvc/heal-results-pvc pvc/heal-data-pvc \
  -n $NS --timeout=60s

# 2. Data generation (only if not already done)
if ! kubectl get job heal-gen-data -n $NS &>/dev/null; then
  echo "[2/4] Generating data (MNIST + Spambase split)..."
  kubectl apply -f k8s/job-gen-data.yaml
  kubectl wait --for=condition=complete job/heal-gen-data -n $NS --timeout=600s
else
  echo "[2/4] Data generation already done, skipping."
fi

# 3. PeerSim events (E2/E3/E4 only)
if [[ "$EXP" != "e1" ]]; then
  if ! kubectl get job heal-peersim -n $NS &>/dev/null; then
    echo "[3/4] Running PeerSim to generate event schedules..."
    kubectl apply -f k8s/job-peersim.yaml
    kubectl wait --for=condition=complete job/heal-peersim -n $NS --timeout=300s
  else
    echo "[3/4] PeerSim already done, skipping."
  fi
else
  echo "[3/4] E1 is crash-free — no PeerSim needed."
fi

# 4. Experiment jobs
echo "[4/4] Launching experiment jobs..."
if [[ "$EXP" == "all" ]]; then
  kubectl apply -f k8s/experiments/
else
  kubectl apply -f k8s/experiments/${EXP}-*.yaml 2>/dev/null || \
    for f in k8s/experiments/${EXP}-*.yaml; do kubectl apply -f "$f"; done
fi

echo ""
echo "=== Deployed. Monitor with: ==="
echo "  kubectl get jobs -n $NS"
echo "  kubectl logs -f job/heal-${EXP}-heal-mnist -n $NS"
echo "  bash k8s/status.sh"
