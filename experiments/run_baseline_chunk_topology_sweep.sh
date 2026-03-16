#!/bin/bash
# Baseline vs chunk-only across topologies and ER p values.
# Same style as hybrid ablation: 2 conditions (baseline, chunk-only), ~5 seeds per graph.
# Graphs: ring, star, grid, fully_connected, d-regular (d=3,10,25), ER (p=0.04,0.08,0.16,0.32).

set -e
cd "$(dirname "$0")/.."
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/baseline_chunk_topology_sweep_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1
ROUNDS=${ROUNDS:-100}
PEERS=${PEERS:-100}
# 5 seeds per graph
SEEDS="${SEEDS:-4235 999 1234 5678 8765}"
BATCH_SIZE=${BATCH_SIZE:-32}
RESULTS_ROOT=${RESULTS_ROOT:-results/cifar100/baseline_chunk_topology_sweep}
EXTRA_ARGS=()
[[ -n "${TIMING:-}" ]] && [[ "${TIMING}" != "0" ]] && EXTRA_ARGS+=(--time-rounds)
[[ -n "${DEVICE:-}" ]] && EXTRA_ARGS+=(--device "$DEVICE")

# Base args (no --seed, no topology-specific flags)
BASE_ARGS=(
  --rounds "$ROUNDS"
  --peers "$PEERS"
  --beta 0.5
  --model "cnn"
  --dataset "cifar100"
  --local-epochs 5
  --mia-attack "baseline"
  --mia-interval 100
  --batch-size "$BATCH_SIZE"
  --mia-baseline-type "loss"
  --partitioner "iid"
  --alpha 0.3
  --weight-decay 0
  --message-type "full"
  --mia-measurement-number 1000
  --no-samples 1000
  --eval-top-k 5
  "${EXTRA_ARGS[@]}"
)

run_one() {
  local topology=$1
  local results_dir=$2
  shift 2
  local extra=("$@")
  python3 -u main.py "${BASE_ARGS[@]}" --topology "$topology" --mia-results-root "$results_dir" "${extra[@]}"
}

echo "=== Baseline vs chunk-only topology sweep ==="
echo "  RESULTS_ROOT=$RESULTS_ROOT  ROUNDS=$ROUNDS  PEERS=$PEERS  BATCH_SIZE=$BATCH_SIZE"
echo "  SEEDS=$SEEDS  TIMING=${TIMING:-0}  DEVICE=${DEVICE:-auto}"
echo ""

# --- Deterministic topologies: ring, star, grid, full ---
for topo in ring star grid full; do
  RESULTS_DIR="${RESULTS_ROOT}/${topo}"
  mkdir -p "$RESULTS_DIR"
  echo "--- topology=$topo  RESULTS_DIR=$RESULTS_DIR ---"
  for seed in $SEEDS; do
    echo "[1/2] Baseline  topo=$topo  seed=$seed"
    run_one "$topo" "$RESULTS_DIR" --seed "$seed"
    echo "[2/2] Chunk only  topo=$topo  seed=$seed"
    run_one "$topo" "$RESULTS_DIR" --seed "$seed" --chunk
    echo ""
  done
done

# --- d-regular: d=3, d=10, d=25 ---
for d in 3 10 25; do
  RESULTS_DIR="${RESULTS_ROOT}/regular_d${d}"
  mkdir -p "$RESULTS_DIR"
  echo "--- topology=regular degree=$d  RESULTS_DIR=$RESULTS_DIR ---"
  for seed in $SEEDS; do
    echo "[1/2] Baseline  regular d=$d  seed=$seed"
    run_one "regular" "$RESULTS_DIR" --seed "$seed" --regular-degree "$d"
    echo "[2/2] Chunk only  regular d=$d  seed=$seed"
    run_one "regular" "$RESULTS_DIR" --seed "$seed" --regular-degree "$d" --chunk
    echo ""
  done
done

# --- ER: p = 0.04, 0.08, 0.16, 0.32 ---
for er_p in 0.04 0.08 0.16 0.32; do
  RESULTS_DIR="${RESULTS_ROOT}/er_p_${er_p}"
  mkdir -p "$RESULTS_DIR"
  echo "--- topology=er er_p=$er_p  RESULTS_DIR=$RESULTS_DIR ---"
  for seed in $SEEDS; do
    echo "[1/2] Baseline  er_p=$er_p  seed=$seed"
    run_one "er" "$RESULTS_DIR" --seed "$seed" --er-p "$er_p"
    echo "[2/2] Chunk only  er_p=$er_p  seed=$seed"
    run_one "er" "$RESULTS_DIR" --seed "$seed" --er-p "$er_p" --chunk
    echo ""
  done
done

echo "=== Done. CSVs in $RESULTS_ROOT/<topo>/ (baseline and chunk-only, 5 seeds each). ==="
echo "Plot per topology: python3 src/plotting/hybrid_privacy_tradeoff.py --results-dir $RESULTS_ROOT/<topo> --out-dir plots/baseline_chunk_<topo>"
