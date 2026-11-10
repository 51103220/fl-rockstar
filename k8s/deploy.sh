#!/usr/bin/env bash
# Deploy the full FL reproduction pipeline to Kyma
# Usage: bash k8s/deploy.sh [e1|e2|e3|e4|all]
#        bash scripts/deploy_cloud.sh deploy [e1|e2|e3|e4|all]
set -euo pipefail

# Resolve repo root regardless of where the script is called from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP=${1:-all}
NS=keda

echo "=== HEAL Reproduction Deploy ==="
echo "Experiment : $EXP"
echo "Namespace  : $NS"
echo ""

# 0. Registry pull secrets
echo "[0/4] Creating pull secrets..."
: "${ARTIFACTORY_USER:?Set ARTIFACTORY_USER}"
: "${ARTIFACTORY_TOKEN:?Set ARTIFACTORY_TOKEN}"
kubectl create secret docker-registry artifactory-pull-secret \
  --docker-server=k8srepo.common.repositories.cloud.sap \
  --docker-username="${ARTIFACTORY_USER}" \
  --docker-password="${ARTIFACTORY_TOKEN}" \
  --namespace=$NS \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret docker-registry dockerhub-pull-secret \
  --docker-server=docker-hub.common.repositories.cloud.sap \
  --docker-username="${ARTIFACTORY_USER}" \
  --docker-password="${ARTIFACTORY_TOKEN}" \
  --namespace=$NS \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl patch serviceaccount default -n $NS \
  -p '{"imagePullSecrets": [{"name": "artifactory-pull-secret"}, {"name": "dockerhub-pull-secret"}]}'

# 1. Storage
echo "[1/4] Applying PVCs..."
kubectl apply -f "${REPO_ROOT}/k8s/pvc.yaml" 
# Note: StorageClass 'files' uses WaitForFirstConsumer — PVCs bind when first pod mounts them

# 2. Data generation (only if not already done)
if ! kubectl get job heal-gen-data -n $NS &>/dev/null; then
  echo "[2/4] Generating data (MNIST + Spambase split)..."
  kubectl apply -f "${REPO_ROOT}/k8s/job-gen-data.yaml"
  kubectl wait --for=condition=complete job/heal-gen-data -n $NS --timeout=600s
else
  echo "[2/4] Data generation already done, skipping."
fi

# 3. PeerSim events (E2/E3/E4 only)
if [[ "$EXP" != "e1" ]]; then
  if ! kubectl get job heal-peersim -n $NS &>/dev/null; then
    echo "[3/4] Running PeerSim to generate event schedules..."
    kubectl apply -f "${REPO_ROOT}/k8s/job-peersim.yaml"
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
  kubectl apply -f "${REPO_ROOT}/k8s/experiments/"
else
  kubectl apply -f "${REPO_ROOT}/k8s/experiments/${EXP}-"*.yaml 2>/dev/null || \
    for f in "${REPO_ROOT}/k8s/experiments/${EXP}-"*.yaml; do kubectl apply -f "$f"; done
fi

echo ""
echo "=== Deployed. Monitor with: ==="
echo "  kubectl get jobs -n $NS"
echo "  kubectl logs -f job/heal-${EXP}-heal-mnist -n $NS"
echo "  bash ${REPO_ROOT}/k8s/status.sh"
