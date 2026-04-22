#!/usr/bin/env bash
# One-command setup for the Mutual Fund Analyzer.
#
# Usage:
#   ./setup.sh                 # create venv, install deps, run full pipeline
#   ./setup.sh --no-run        # only install, don't run
#   ./setup.sh --offline       # use synthetic data (no network)

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
RUN_PIPELINE=1
EXTRA_ARGS=()

for arg in "$@"; do
    case $arg in
        --no-run)   RUN_PIPELINE=0 ;;
        --offline)  EXTRA_ARGS+=("--offline") ;;
        *)          EXTRA_ARGS+=("$arg") ;;
    esac
done

echo "▶ Checking Python ..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "✖ Python not found. Please install Python 3.10+."; exit 1
fi
"$PYTHON_BIN" --version

echo "▶ Creating virtual environment at $VENV_DIR ..."
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "▶ Upgrading pip ..."
pip install --upgrade pip --quiet

echo "▶ Installing dependencies ..."
pip install -r requirements.txt --quiet

echo "▶ Creating data directories ..."
mkdir -p data/raw data/processed

if [ "$RUN_PIPELINE" -eq 1 ]; then
    echo "▶ Running the full pipeline with default profile ..."
    echo "   (amount=₹5,00,000  horizon=5y  risk=Medium)"
    python main.py --amount 500000 --horizon 5 --risk Medium "${EXTRA_ARGS[@]}"
else
    echo "▶ Setup complete. Activate the venv and run:"
    echo "     source $VENV_DIR/bin/activate"
    echo "     python main.py"
fi

echo "✓ Done."
