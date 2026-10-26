#!/usr/bin/env bash
# Show status of all experiment jobs
NS=heal-repro
echo "=== HEAL Experiment Status ==="
kubectl get jobs -n $NS \
  -o custom-columns="JOB:.metadata.name,STATUS:.status.conditions[0].type,ACTIVE:.status.active,SUCCEEDED:.status.succeeded,FAILED:.status.failed" \
  --sort-by=.metadata.name
echo ""
DONE=$(kubectl get jobs -n $NS -o json | python3 -c "
import json,sys
jobs=json.load(sys.stdin)['items']
done=sum(1 for j in jobs if j.get('status',{}).get('succeeded',0)>0)
total=len(jobs)
print(f'Progress: {done}/{total} jobs completed')
")
echo "$DONE"
