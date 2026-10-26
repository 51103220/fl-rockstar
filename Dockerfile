FROM python:3.11-slim

WORKDIR /workspace

# System deps for networkx, matplotlib, torchvision
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Data directory (mounted as volume in docker-compose)
RUN mkdir -p /data/cifar10 /workspace/results

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/run_experiments.py"]
