#!/usr/bin/env bash
# Deploy to GCP Cloud Run Jobs (CPU) or GKE (GPU)
# Requires: gcloud CLI, docker
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-central1}"
IMAGE="gcr.io/${PROJECT_ID}/fl-rockstar:latest"
JOB_NAME="fl-rockstar-train"

case "${1:-help}" in
  build-push)
    echo "Building GPU image and pushing to GCR..."
    docker build -f docker/Dockerfile.gpu -t "${IMAGE}" .
    docker push "${IMAGE}"
    ;;
  deploy-cpu)
    echo "Deploying to Cloud Run Jobs (CPU)..."
    gcloud run jobs create "${JOB_NAME}" \
      --image="${IMAGE}" \
      --region="${REGION}" \
      --memory=8Gi \
      --cpu=4 \
      --max-retries=1 \
      --task-timeout=3h \
      --set-env-vars="PYTHONUNBUFFERED=1,PYTHONPATH=/workspace" \
      --project="${PROJECT_ID}"
    ;;
  run-job)
    echo "Executing Cloud Run Job..."
    gcloud run jobs execute "${JOB_NAME}" \
      --region="${REGION}" \
      --project="${PROJECT_ID}" \
      --wait
    ;;
  deploy-gke-gpu)
    echo "Apply GKE GPU job manifest..."
    kubectl apply -f docker/k8s_job_gpu.yaml
    ;;
  logs)
    gcloud logging read \
      "resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}" \
      --limit=100 \
      --format="value(textPayload)" \
      --project="${PROJECT_ID}"
    ;;
  *)
    echo "Usage: $0 {build-push|deploy-cpu|run-job|deploy-gke-gpu|logs}"
    exit 1
    ;;
esac
