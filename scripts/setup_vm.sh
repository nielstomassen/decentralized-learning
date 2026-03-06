#!/usr/bin/env bash
# One-time VM setup: venv + deps. Use --gpu to install PyTorch with CUDA.
set -e
cd "$(dirname "$0")/.."

USE_GPU=false
for arg in "$@"; do
  case "$arg" in
    --gpu) USE_GPU=true ;;
  esac
done

if [[ ! -d .venv ]]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi
echo "Activate with: source .venv/bin/activate"
# Allow this script to run without sourcing (we call pip via .venv)
PIP=".venv/bin/pip"
PYTHON=".venv/bin/python3"
if [[ ! -x "$PIP" ]]; then
  PIP="pip"
  PYTHON="python3"
  echo "Using system pip/python (activate .venv first for isolation)"
fi

"$PIP" install --upgrade pip

if "$USE_GPU"; then
  echo "Installing PyTorch with CUDA 12.1..."
  "$PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

echo "Installing requirements.txt..."
"$PIP" install -r requirements.txt

echo ""
echo "Setup done. Next:"
echo "  source .venv/bin/activate"
echo "  ./experiments/run_hybrid_ablation.sh"
if "$USE_GPU"; then
  echo ""
  echo "GPU: use BATCH_SIZE=32 or 64 and optionally TIMING=1 CUDA_BENCHMARK=1 for faster runs."
fi
