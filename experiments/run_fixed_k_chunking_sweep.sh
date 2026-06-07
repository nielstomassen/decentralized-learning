#!/bin/bash
# Fixed-K chunking sweep: K ∈ {8, 16, 32, 64, 128} × er_p × seeds (no DP).
# Output: results/fixed_k_chunking_sweep/<K>_standard_chunking_sweep/er_p_<p>/
# Same non-chunking hyperparameters as run_hybrid_ablation.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fixed_k_chunking_sweep_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1
ROUNDS=${ROUNDS:-100}
PEERS=${PEERS:-100}
SEEDS="${SEEDS:-4235 45 929 9838 12}"
ER_PS="${ER_PS:-0.08 0.16}"
K_VALUES="${K_VALUES:-8 16 32 64 128}"
RESULTS_ROOT=${RESULTS_ROOT:-results/fixed_k_chunking_sweep}
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

echo "=== Fixed-K chunking sweep (no DP) — K × er_p × seeds ==="
echo "  RESULTS_ROOT=$RESULTS_ROOT  ROUNDS=$ROUNDS  PEERS=$PEERS  BATCH_SIZE=$BATCH_SIZE"
echo "  K_VALUES=$K_VALUES  ER_PS=$ER_PS  SEEDS=$SEEDS  TIMING=${TIMING:-0}  DEVICE=${DEVICE:-auto}"
echo ""

total=$(( $(echo $K_VALUES | wc -w) * $(echo $ER_PS | wc -w) * $(echo $SEEDS | wc -w) ))
n=0
for global_k in $K_VALUES; do
  for er_p in $ER_PS; do
    RESULTS_DIR="${RESULTS_ROOT}/${global_k}_standard_chunking_sweep/er_p_${er_p}"
    mkdir -p "$RESULTS_DIR"
    echo "--- K=$global_k  er_p=$er_p  RESULTS_DIR=$RESULTS_DIR ---"

    for seed in $SEEDS; do
      n=$((n + 1))
      echo "[$n/$total] standard_chunking  K=$global_k  er_p=$er_p  seed=$seed"
      python3 -u main.py "${BASE_ARGS[@]}" \
        --er-p "$er_p" \
        --mia-results-root "$RESULTS_DIR" \
        --seed "$seed" \
        --chunk \
        --chunking-mode standard_chunking \
        --standard-chunking-global-k "$global_k"
      echo ""
    done
  done
done

echo "=== Done. CSVs in $RESULTS_ROOT/<K>_standard_chunking_sweep/er_p_<p>/ (one run per seed per K × er_p). ==="
echo "Use as fixed-K refs in hybrid plots (--additional-results-dir or auto via --er-p in analyze_hybrid_noise_clip_sweep.py)."
echo "  See run_hybrid_ablation.sh / README for full hybrid_privacy_tradeoff command."
