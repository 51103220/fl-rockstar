# Kyma Operations Reference

Namespace: `keda`

---

## Deploy

```bash
# Full deploy (namespace → PVCs → gen-data → peersim → all 40 jobs)
bash k8s/deploy.sh

# Single experiment only
bash k8s/deploy.sh e1
```

---

## Health Checks

### Overall job progress
```bash
kubectl get jobs -n keda \
  -o custom-columns="JOB:.metadata.name,ACTIVE:.status.active,SUCCEEDED:.status.succeeded,FAILED:.status.failed,START:.status.startTime" \
  --sort-by=.metadata.name
```

### Quick summary (done / total)
```bash
bash k8s/status.sh
```

### Jobs currently running
```bash
kubectl get jobs -n keda --field-selector=status.active=1
```

### Failed jobs
```bash
kubectl get jobs -n keda -o json \
  | python3 -c "
import json, sys
jobs = json.load(sys.stdin)['items']
failed = [j['metadata']['name'] for j in jobs if j.get('status', {}).get('failed', 0) > 0]
print('\n'.join(failed) if failed else 'No failed jobs')
"
```

### Pod status (all experiment pods)
```bash
kubectl get pods -n keda --sort-by=.status.phase
```

### Pods not in Running/Completed state
```bash
kubectl get pods -n keda --field-selector='status.phase!=Running,status.phase!=Succeeded'
```

### Node resource pressure
```bash
kubectl top nodes
kubectl top pods -n keda --sort-by=cpu
```

### PVC usage
```bash
kubectl get pvc -n keda
```

---

## Logs

### Logs for a specific job  
Pattern: `heal-{exp}-{algo}-{task}`  — e.g. `heal-e1-heal-mnist`

```bash
# Follow live
kubectl logs -f job/heal-e1-heal-mnist -n keda

# Last 100 lines
kubectl logs job/heal-e1-heal-mnist -n keda --tail=100
```

### Logs for a specific pod
```bash
# Find the pod name first
kubectl get pods -n keda -l job-name=heal-e1-heal-mnist

# Then tail it
kubectl logs -f <pod-name> -n keda
```

### Logs from a failed pod (previous container)
```bash
kubectl logs <pod-name> -n keda --previous
```

### Stream logs from all running pods (requires stern)
```bash
stern -n keda "heal-" --tail=20
```

### Dump logs from all jobs to local files
```bash
mkdir -p logs
for job in $(kubectl get jobs -n keda -o jsonpath='{.items[*].metadata.name}'); do
  kubectl logs job/$job -n keda > logs/$job.log 2>&1
  echo "Saved logs/$job.log"
done
```

---

## Mid-Run Inspection

### Describe a job (events, restart count, conditions)
```bash
kubectl describe job heal-e3-heal-mnist -n keda
```

### Describe a pod (useful for OOM / scheduling issues)
```bash
kubectl describe pod <pod-name> -n keda
```

### Check if results are being written to PVC
```bash
# Spin up a debug pod to inspect the shared volume
kubectl run pvc-inspector --rm -it --restart=Never \
  --image=busybox -n keda \
  --overrides='{"spec":{"volumes":[{"name":"r","persistentVolumeClaim":{"claimName":"heal-results-pvc"}}],"containers":[{"name":"pvc-inspector","image":"busybox","command":["sh"],"stdin":true,"tty":true,"volumeMounts":[{"mountPath":"/results","name":"r"}]}]}}' \
  -- sh

# Inside the pod:
# ls /results/
# find /results -name "results.json" | wc -l
```

### Count result files written so far
```bash
kubectl run result-check --rm -it --restart=Never \
  --image=busybox -n keda \
  --overrides='{"spec":{"volumes":[{"name":"r","persistentVolumeClaim":{"claimName":"heal-results-pvc"}}],"containers":[{"name":"c","image":"busybox","command":["find","/results","-name","results.json"],"volumeMounts":[{"mountPath":"/results","name":"r"}]}]}}' \
  2>/dev/null
```

---

## Collect Results

```bash
# Pull all results.json from the PVC to local results/
bash k8s/collect_results.sh

# Then plot
source .venv/bin/activate
python scripts/plot_results.py --results-dir results/ --out-dir results/plots/
```

---

## Troubleshooting

| Symptom | Command |
|---|---|
| Job stuck in Pending | `kubectl describe pod <pod> -n keda` → check Events for scheduling errors |
| OOMKilled | `kubectl describe pod <pod> -n keda` → look for `OOMKilled`; increase memory limit in YAML |
| ImagePullBackOff | `kubectl describe pod <pod> -n keda` → check image name / registry access |
| PVC not bound | `kubectl describe pvc -n keda` → check StorageClass availability |
| Job failed, want to retry | `kubectl delete job <name> -n keda && kubectl apply -f k8s/experiments/<name>.yaml` |

---

## Teardown

```bash
# Delete all experiment jobs (keep PVCs / data)
kubectl delete jobs -n keda --all

# Full teardown (deletes everything including PVCs)
bash k8s/teardown.sh
```
