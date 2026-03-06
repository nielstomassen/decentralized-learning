#!/bin/bash
# Run only the hybrid condition (DP + chunk) from the hybrid ablation.
# Same seeds and ER p values as run_hybrid_ablation.sh; writes into the same RESULTS_ROOT
# so you can re-plot with hybrid_privacy_tradeoff.py together with existing baseline/DP-only/chunk-only CSVs.

set -e
cd "$(dirname "$0")/.."

ROUNDS=${ROUNDS:-30}
PEERS=${PEERS:-100}
SEEDS="${SEEDS:-4235}"
ER_PS="${ER_PS:-0.08}"
DP_NOISE=${DP_NOISE:-0.5}
RESULTS_ROOT=${RESULTS_ROOT:-results/wookie}

BASE_ARGS=(
  --rounds "$ROUNDS"
  --peers "$PEERS"
  --topology "er"
  --beta 0.7
  --model "cnn"
  --dataset "emnist"
  --local-epochs 5
  --mia-attack "baseline"
  --mia-interval 30
  --batch-size 16
  --mia-baseline-type "loss"
  --partitioner "dirichlet"
  --alpha 0.3
  --weight-decay 0
  --message-type "full"
  --mia-measurement-number 1000
  --no-samples 500
  --eval-top-k 1
)

echo "=== Hybrid only (DP + chunk) over ER p values and seeds ==="
echo "  RESULTS_ROOT=$RESULTS_ROOT  ROUNDS=$ROUNDS  PEERS=$PEERS  DP_NOISE=$DP_NOISE"
echo "  ER_PS=$ER_PS  SEEDS=$SEEDS"
echo ""

for er_p in $ER_PS; do
  RESULTS_DIR="${RESULTS_ROOT}/er_p_${er_p}"
  mkdir -p "$RESULTS_DIR"
  echo "--- er_p=$er_p  RESULTS_DIR=$RESULTS_DIR ---"

  for seed in $SEEDS; do
    echo "Hybrid (DP + chunk)  er_p=$er_p  seed=$seed"
    python3 main.py "${BASE_ARGS[@]}" --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed" --dp --dp-noise "$DP_NOISE" --chunk
    echo ""
  done
done

echo "=== Done. Hybrid CSVs in $RESULTS_ROOT/er_p_<p>. ==="
echo "Plot: python3 src/plotting/hybrid_privacy_tradeoff.py --results-dir $RESULTS_ROOT/er_p_<p> --out-dir plots/hybrid"
