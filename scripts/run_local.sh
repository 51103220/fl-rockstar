#!/usr/bin/env bash
# Build and run locally (CPU, dev mode)
set -euo pipefail

COMPOSE="docker compose"

case "${1:-help}" in
  build)
    $COMPOSE build
    ;;
  jupyter)
    echo "Starting Jupyter on http://localhost:8888"
    $COMPOSE --profile dev up jupyter
    ;;
  train)
    echo "Running full experiment suite (E1–E4)..."
    $COMPOSE --profile train run --rm trainer
    ;;
  test)
    echo "Running unit tests inside container..."
    $COMPOSE run --rm --no-deps trainer \
      python -m pytest tests/ -v --tb=short
    ;;
  shell)
    $COMPOSE run --rm trainer bash
    ;;
  down)
    $COMPOSE down --remove-orphans
    ;;
  *)
    echo "Usage: $0 {build|jupyter|train|test|shell|down}"
    exit 1
    ;;
esac
