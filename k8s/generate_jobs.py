#!/usr/bin/env python3
"""Generate all 40 experiment Job YAML files."""
import os

ALGOS = ["heal", "fl", "gossip", "epidemic", "epidemic-ring"]
EXPS  = ["e1", "e2", "e3", "e4"]
TASKS = ["mnist", "spambase"]

TEMPLATE = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: heal-{exp}-{algo}-{task}
  namespace: heal-repro
  labels:
    exp: "{exp}"
    algo: "{algo}"
    task: "{task}"
spec:
  backoffLimit: 5
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: experiment
        image: python:3.11-slim
        command:
        - bash
        - -c
        - |
          pip install torch torchvision datasets networkx numpy scipy \\
            matplotlib seaborn pandas tqdm pyyaml flwr gossipy --quiet
          git clone https://github.com/51103220/fl-rockstar.git /app
          cd /app
          python scripts/run_experiment.py \\
            --exp {exp} \\
            --algo {algo} \\
            --task {task} \\
            --n-nodes 100 \\
            --n-cycles 1000 \\
            --seed 42 \\
            --data-dir /data \\
            --ckpt-dir /checkpoints/{exp}/{algo}/{task} \\
            --results-dir /results
        resources:
          requests:
            cpu: "{cpu}"
            memory: "{mem}"
          limits:
            cpu: "{cpu}"
            memory: "{mem}"
        volumeMounts:
        - name: data
          mountPath: /data
          readOnly: true
        - name: checkpoints
          mountPath: /checkpoints
        - name: results
          mountPath: /results
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: heal-data-pvc
      - name: checkpoints
        persistentVolumeClaim:
          claimName: heal-checkpoints-pvc
      - name: results
        persistentVolumeClaim:
          claimName: heal-results-pvc
"""

# CPU/mem per algo (FL needs more for central server)
RESOURCES = {
    "fl":           ("8", "16Gi"),
    "heal":         ("8", "16Gi"),
    "gossip":       ("4", "8Gi"),
    "epidemic":     ("4", "8Gi"),
    "epidemic-ring":("4", "8Gi"),
}

out_dir = os.path.join(os.path.dirname(__file__), "experiments")
os.makedirs(out_dir, exist_ok=True)

for exp in EXPS:
    for algo in ALGOS:
        for task in TASKS:
            cpu, mem = RESOURCES[algo]
            content = TEMPLATE.format(
                exp=exp, algo=algo, task=task, cpu=cpu, mem=mem
            )
            fname = f"{exp}-{algo}-{task}.yaml"
            with open(os.path.join(out_dir, fname), "w") as f:
                f.write(content)

print(f"Generated {len(EXPS) * len(ALGOS) * len(TASKS)} job files in k8s/experiments/")
