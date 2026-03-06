#!/bin/bash
# 2×2 ablation: (no DP, no chunk), (DP only), (chunk only), (DP + chunk).
# Runs each condition for every seed in SEEDS; plot script averages over seeds.

set -e
cd "$(dirname "$0")/.."

ROUNDS=${ROUNDS:-60}
PEERS=${PEERS:-100}
# Space-separated list of seeds (each condition is run once per seed)
SEEDS="${SEEDS:-4235}"
# Space-separated list of ER graph p values to sweep
ER_PS="${ER_PS:-0.08}"
DP_NOISE=${DP_NOISE:-0.5}
RESULTS_ROOT=${RESULTS_ROOT:-results/cifar100/hybrid_ablation}
# GPU-friendly: larger batch (e.g. 32 or 64), set TIMING=1 to time rounds, DEVICE=cuda:0 to force GPU
BATCH_SIZE=${BATCH_SIZE:-16}
EXTRA_ARGS=()
[[ -n "${TIMING:-}" ]] && [[ "${TIMING}" != "0" ]] && EXTRA_ARGS+=(--time-rounds)
[[ -n "${DEVICE:-}" ]] && EXTRA_ARGS+=(--device "$DEVICE")

# Base args without --seed or --er-p (we pass those per run)
BASE_ARGS=(
  --rounds "$ROUNDS"
  --peers "$PEERS"
  --topology "er"
  --beta 0.7
  --model "resnet56"
  --dataset "cifar100"
  --local-epochs 5
  --mia-attack "baseline"
  --mia-interval 60
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

echo "=== Hybrid ablation: 2×2 (DP × chunk) over ER p values and seeds ==="
echo "  RESULTS_ROOT=$RESULTS_ROOT  ROUNDS=$ROUNDS  PEERS=$PEERS  DP_NOISE=$DP_NOISE  BATCH_SIZE=$BATCH_SIZE"
echo "  ER_PS=$ER_PS  SEEDS=$SEEDS  TIMING=${TIMING:-0}  DEVICE=${DEVICE:-auto}"
echo ""

for er_p in $ER_PS; do
  RESULTS_DIR="${RESULTS_ROOT}/er_p_${er_p}"
  mkdir -p "$RESULTS_DIR"
  echo "--- er_p=$er_p  RESULTS_DIR=$RESULTS_DIR ---"

  for seed in $SEEDS; do
    # 1) No DP, no chunk (baseline)
    echo "[1/4] No DP, no chunk (baseline)  er_p=$er_p  seed=$seed"
    python3 main.py "${BASE_ARGS[@]}" --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed"
    echo ""

    # 2) DP only
    echo "[2/4] DP only  er_p=$er_p  seed=$seed"
    python3 main.py "${BASE_ARGS[@]}" --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed" --dp --dp-noise "$DP_NOISE"
    echo ""

    # 3) Chunk only
    echo "[3/4] Chunk only  er_p=$er_p  seed=$seed"
    python3 main.py "${BASE_ARGS[@]}" --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed" --chunk
    echo ""

    # 4) DP + chunk (hybrid)
    echo "[4/4] DP + chunk (hybrid)  er_p=$er_p  seed=$seed"
    python3 main.py "${BASE_ARGS[@]}" --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed" --dp --dp-noise "$DP_NOISE" --chunk
    echo ""
  done
done

echo "=== Done. CSVs in $RESULTS_ROOT/er_p_<p> (one file per condition per seed per er_p). ==="
echo "Plot per er_p (averages over seeds): python3 src/plotting/hybrid_privacy_tradeoff.py --results-dir $RESULTS_ROOT/er_p_<p> --out-dir plots/hybrid"
