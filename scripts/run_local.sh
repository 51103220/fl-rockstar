#!/usr/bin/env bash
# run_local.sh — set up venv and run quick local experiments
# Usage:
#   bash scripts/run_local.sh                          # runs all quick tests
#   bash scripts/run_local.sh --exp e1 --algo heal     # single run
#   bash scripts/run_local.sh --setup-only             # install only, no run
#   bash scripts/run_local.sh --plot-only              # plot from existing results
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON=""

# ─────────────────────────── helpers ─────────────────────────────────────────

log()  { echo "[run_local] $*"; }
die()  { echo "[run_local] ERROR: $*" >&2; exit 1; }

find_python() {
    for candidate in python3.11 python3.10 python3.9 python3 python; do
        if command -v "$candidate" &>/dev/null; then
            ver=$("$candidate" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
            # require >= (3, 9)
            ok=$("$candidate" -c \
                "import sys; print(sys.version_info >= (3,9))" 2>/dev/null)
            if [[ "$ok" == "True" ]]; then
                echo "$candidate"
                return
            fi
        fi
    done
    die "Python 3.9+ not found. Install it first."
}

# ─────────────────────────── venv setup ──────────────────────────────────────

setup_venv() {
    PYTHON=$(find_python)
    log "Using $PYTHON ($(${PYTHON} --version))"

    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtual environment at $VENV_DIR ..."
        "$PYTHON" -m venv "$VENV_DIR"
    else
        log "Virtual environment already exists at $VENV_DIR"
    fi

    # Activate
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON="$VENV_DIR/bin/python"

    log "Installing/updating dependencies..."
    pip install --quiet --upgrade pip

    # Core deps from requirements.txt
    pip install --quiet -r "$REPO_ROOT/requirements.txt"

    # Extra deps for full reproduction (not in base requirements.txt)
    pip install --quiet scikit-learn matplotlib seaborn pandas

    # Optional: Flower for FL baseline (skip if install fails)
    pip install --quiet flwr 2>/dev/null || \
        log "  [warn] flwr not installed — FL baseline will use built-in FedAvg"

    log "Setup complete."
}

# ─────────────────────────── argument parsing ─────────────────────────────────

SETUP_ONLY=false
PLOT_ONLY=false
RUN_ALL=true
EXP=""
ALGO=""
TASK="mnist"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-only) SETUP_ONLY=true; RUN_ALL=false; shift ;;
        --plot-only)  PLOT_ONLY=true;  RUN_ALL=false; shift ;;
        --exp)  EXP="$2";  RUN_ALL=false; shift 2 ;;
        --algo) ALGO="$2"; RUN_ALL=false; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS="$*"; break ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ─────────────────────────── main ────────────────────────────────────────────

cd "$REPO_ROOT"

setup_venv

if $SETUP_ONLY; then
    log "Setup done. Activate with: source $VENV_DIR/bin/activate"
    exit 0
fi

if $PLOT_ONLY; then
    log "Plotting results from ./results ..."
    "$PYTHON" scripts/plot_results.py --results-dir ./results
    log "Plots saved to ./results/plots/"
    exit 0
fi

run_quick() {
    local exp=$1
    local algo=$2
    local task=${3:-mnist}
    log "Running: exp=$exp algo=$algo task=$task (quick mode)"
    "$PYTHON" scripts/run_experiment.py \
        --exp "$exp" --algo "$algo" --task "$task" \
        --quick --local-epochs 1 \
        ${EXTRA_ARGS}
    echo ""
}

if [[ -n "$EXP" && -n "$ALGO" ]]; then
    # Single specified run
    run_quick "$EXP" "$ALGO" "$TASK"
elif $RUN_ALL; then
    # Full quick suite: all algorithms × E1 + E2, both tasks
    log "Running full quick suite (10 nodes, 30 cycles each)..."
    log "This takes ~3-5 min per run. HEAL and FL are slowest."
    echo ""

    for task in mnist spambase; do
        log "=== Task: $task ==="
        run_quick e1 heal        "$task"
        run_quick e1 fl          "$task"
        run_quick e1 gossip      "$task"
        run_quick e1 epidemic    "$task"
        run_quick e1 epidemic-ring "$task"
        run_quick e2 heal        "$task"  # crash test
        run_quick e3 heal        "$task"  # hub crash + re-election
    done
fi

# Plot whatever results exist
log "Generating plots..."
"$PYTHON" scripts/plot_results.py --results-dir ./results 2>/dev/null || \
    log "  [skip] No results yet or matplotlib missing."

log ""
log "Done. Results in ./results/  |  Plots in ./results/plots/"
log ""
log "To activate the venv manually:"
log "  source $VENV_DIR/bin/activate"
log ""
log "Full Kyma run (100 nodes, 1000 cycles):"
log "  bash k8s/deploy.sh"
