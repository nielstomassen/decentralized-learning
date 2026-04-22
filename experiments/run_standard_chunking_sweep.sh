#!/bin/bash
# Standard chunking baseline only (node-level global K partitions, same subset to all neighbors).
# Maps er_p -> global K: 0.08 -> 8, 0.16 -> 16 (other er_p: warn, K=8).
# Outer loop: seeds; inner loop: ER p. Same non-chunking hyperparameters as run_hybrid_ablation.sh chunk-only arm.

set -e
cd "$(dirname "$0")/.."
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/standard_chunking_sweep_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1
ROUNDS=${ROUNDS:-100}
PEERS=${PEERS:-100}
SEEDS="${SEEDS:-4235 45 929 9838 12}"
ER_PS="${ER_PS:-0.08 0.16}"
RESULTS_ROOT=${RESULTS_ROOT:-results/cifar100/standard_chunking_sweep}
BATCH_SIZE=${BATCH_SIZE:-32}
EXTRA_ARGS=()
[[ -n "${TIMING:-}" ]] && [[ "${TIMING}" != "0" ]] && EXTRA_ARGS+=(--time-rounds)
[[ -n "${DEVICE:-}" ]] && EXTRA_ARGS+=(--device "$DEVICE")

BASE_ARGS=(
  --rounds "$ROUNDS"
  --peers "$PEERS"
  --topology "er"
  --beta 0.5
  --model "cnn"
  --dataset "cifar100"
  --local-epochs 5
  --mia-attack "baseline"
  --mia-interval 100
  --batch-size "$BATCH_SIZE"
  --dp-logical-batch-size "$BATCH_SIZE"
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

echo "=== Standard chunking sweep (no DP) — seeds outer, er_p inner ==="
echo "  RESULTS_ROOT=$RESULTS_ROOT  ROUNDS=$ROUNDS  PEERS=$PEERS  BATCH_SIZE=$BATCH_SIZE"
echo "  ER_PS=$ER_PS  SEEDS=$SEEDS  TIMING=${TIMING:-0}  DEVICE=${DEVICE:-auto}"
echo ""

for seed in $SEEDS; do
  echo "======== seed=$seed ========"

  for er_p in $ER_PS; do
    case "$er_p" in
      0.08) GLOBAL_K=8 ;;
      0.16) GLOBAL_K=16 ;;
      *)
        echo "WARN: er_p=$er_p not in {0.08,0.16}; using GLOBAL_K=8"
        GLOBAL_K=8
        ;;
    esac
    RESULTS_DIR="${RESULTS_ROOT}/er_p_${er_p}"
    mkdir -p "$RESULTS_DIR"
    echo "[standard_chunking] er_p=$er_p  global_k=$GLOBAL_K  seed=$seed  -> $RESULTS_DIR"
    python3 -u main.py "${BASE_ARGS[@]}" \
      --er-p "$er_p" \
      --mia-results-root "$RESULTS_DIR" \
      --seed "$seed" \
      --chunk \
      --chunking-mode standard_chunking \
      --standard-chunking-global-k "$GLOBAL_K"
    echo ""
  done
done

echo "=== Done. CSVs in $RESULTS_ROOT/er_p_<p>/ (one run per seed per er_p). ==="
echo "Plot (per er_p, averages over seeds): python3 src/plotting/hybrid_privacy_tradeoff.py --results-dir $RESULTS_ROOT/er_p_<p> --out-dir plots/hybrid"
