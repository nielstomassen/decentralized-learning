#!/bin/bash
# Run only the hybrid condition (DP + chunk) with same settings as run_hybrid_ablation.sh,
# but sweep over noise multiplier and max grad norm to find a sweet spot.
# Output CSVs go to RESULTS_ROOT/er_p_<p>/; analyze with analyze_hybrid_noise_clip_sweep.py

set -e
cd "$(dirname "$0")/.."
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/hybrid_noise_clip_sweep_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1
ROUNDS=${ROUNDS:-100}
PEERS=${PEERS:-100}
SEEDS="${SEEDS:-999}"
ER_PS="${ER_PS:-0.08}"
RESULTS_ROOT=${RESULTS_ROOT:-results/hybrid_noise_clip_sweep}
BATCH_SIZE=${BATCH_SIZE:-32}
# Sweep values (space-separated). Tune these to find sweet spot.
NOISE_VALUES="${NOISE_VALUES:-0.3 0.4 0.5 0.6}"
CLIP_VALUES="${CLIP_VALUES:-1.0 1.5 2.0}"
EXTRA_ARGS=()
[[ -n "${TIMING:-}" ]] && [[ "${TIMING}" != "0" ]] && EXTRA_ARGS+=(--time-rounds)
[[ -n "${DEVICE:-}" ]] && EXTRA_ARGS+=(--device "$DEVICE")

# Same base args as run_hybrid_ablation.sh; DP runs use physical batch 12, logical BATCH_SIZE
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

echo "=== Hybrid-only sweep: noise × max_grad_norm (same settings as hybrid ablation) ==="
echo "  RESULTS_ROOT=$RESULTS_ROOT  ROUNDS=$ROUNDS  PEERS=$PEERS  BATCH_SIZE=$BATCH_SIZE"
echo "  NOISE_VALUES=$NOISE_VALUES  CLIP_VALUES=$CLIP_VALUES"
echo "  ER_PS=$ER_PS  SEEDS=$SEEDS  TIMING=${TIMING:-0}  DEVICE=${DEVICE:-auto}"
echo ""

total=$(( $(echo $NOISE_VALUES | wc -w) * $(echo $CLIP_VALUES | wc -w) * $(echo $SEEDS | wc -w) * $(echo $ER_PS | wc -w) ))
n=0
for er_p in $ER_PS; do
  RESULTS_DIR="${RESULTS_ROOT}/er_p_${er_p}"
  mkdir -p "$RESULTS_DIR"
  echo "--- er_p=$er_p  RESULTS_DIR=$RESULTS_DIR ---"

  for noise in $NOISE_VALUES; do
    for clip in $CLIP_VALUES; do
      for seed in $SEEDS; do
        n=$((n + 1))
        echo "[$n/$total] Hybrid  noise=$noise  clip=$clip  er_p=$er_p  seed=$seed"
        python3 -u main.py "${BASE_ARGS[@]}" --batch-size 12 --er-p "$er_p" \
          --mia-results-root "$RESULTS_DIR" --seed "$seed" \
          --dp --dp-noise "$noise" --dp-clip "$clip" --chunk
        echo ""
      done
    done
  done
done

echo "=== Done. CSVs in $RESULTS_ROOT/er_p_<p>/ (one file per noise/clip/seed). ==="
echo "Analyze: python3 scripts/analyze_hybrid_noise_clip_sweep.py --results-dir $RESULTS_ROOT/er_p_0.08 --out-dir plots/hybrid_noise_clip_sweep"
