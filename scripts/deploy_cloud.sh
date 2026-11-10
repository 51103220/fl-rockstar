set -euo pipefail

# Always resolve paths relative to repo root, not scripts/
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REGISTRY="k8srepo.common.repositories.cloud.sap"
IMAGE="${REGISTRY}/fl-rockstar:latest"
NS="keda"
EXP="${2:-all}"

case "${1:-help}" in
  build-push)
    echo "Building image and pushing to Artifactory..."
    : "${ARTIFACTORY_USER:?Set ARTIFACTORY_USER}"
    : "${ARTIFACTORY_TOKEN:?Set ARTIFACTORY_TOKEN}"
    echo "${ARTIFACTORY_TOKEN}" | docker login k8srepo.common.repositories.cloud.sap \
      --username "${ARTIFACTORY_USER}" --password-stdin
    docker build -f "${REPO_ROOT}/Dockerfile" -t "${IMAGE}" "${REPO_ROOT}"
    docker push "${IMAGE}"
    ;;

  deploy)
    echo "Deploying to Kyma namespace: ${NS} (exp=${EXP})..."
    bash "${REPO_ROOT}/k8s/deploy.sh" "${EXP}"
    ;;

  status)
    echo "=== Job status in ${NS} ==="
    kubectl get jobs -n "${NS}" \
      -o custom-columns="JOB:.metadata.name,ACTIVE:.status.active,SUCCEEDED:.status.succeeded,FAILED:.status.failed" \
      --sort-by=.metadata.name
    ;;

  logs)
    JOB="${3:?Usage: $0 logs <job-name>}"
    kubectl logs -f "job/${JOB}" -n "${NS}"
    ;;

  teardown)
    echo "Tearing down namespace ${NS}..."
    bash "${REPO_ROOT}/k8s/teardown.sh"
    ;;

  *)
    echo "Usage: $0 {build-push|deploy [exp]|status|logs <job-name>|teardown}"
    echo ""
    echo "  build-push          Build Docker image and push to Artifactory"
    echo "  deploy [e1|e2|all]  Deploy experiment jobs to Kyma (default: all)"
    echo "  status              Show job status"
    echo "  logs <job-name>     Stream logs from a job"
    echo "  teardown            Delete all resources in namespace"
    exit 1
    ;;
esac
