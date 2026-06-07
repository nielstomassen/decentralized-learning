#!/bin/bash
# 2×2 ablation: (no DP, no chunk), (DP only), (chunk only), (DP + chunk).
# Runs each condition for every seed in SEEDS; plot script averages over seeds.
# Used for ChunkDP ablation section in the thesis
set -e
cd "$(dirname "$0")/.."
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/hybrid_ablation_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1
ROUNDS=${ROUNDS:-100}
PEERS=${PEERS:-100}
# Space-separated list of seeds (each condition is run once per seed)
SEEDS="${SEEDS:-4235 45 929 9838 12}"
# Space-separated list of ER graph p values to sweep
ER_PS="${ER_PS:-0.16 0.08}"
DP_NOISE=${DP_NOISE:-0.5}
RESULTS_ROOT=${RESULTS_ROOT:-results/hybrid_ablation}
# Single batch size for ablation (32). DP runs override to 12 for GPU memory and use Opacus BatchMemoryManager with logical BATCH_SIZE.
BATCH_SIZE=${BATCH_SIZE:-32}
EXTRA_ARGS=()
[[ -n "${TIMING:-}" ]] && [[ "${TIMING}" != "0" ]] && EXTRA_ARGS+=(--time-rounds)
[[ -n "${DEVICE:-}" ]] && EXTRA_ARGS+=(--device "$DEVICE")

# Base args without --seed or --er-p (we pass those per run)
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
   python3 -u main.py "${BASE_ARGS[@]}" --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed"
   echo ""

    # 2) DP only — physical batch 12 for GPU memory, logical batch BATCH_SIZE
   echo "[2/4] DP only  er_p=$er_p  seed=$seed"
   python3 -u main.py "${BASE_ARGS[@]}" --batch-size 12 --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed" --dp --dp-noise "$DP_NOISE" || { echo "ERROR: [2/4] DP-only run failed with exit code $?"; exit 1; }
   echo ""
    # 3) Chunk only
    echo "[3/4] Chunk only  er_p=$er_p  seed=$seed"
    python3 -u main.py "${BASE_ARGS[@]}" --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed" --chunk --chunking-mode "topology_rowblocks" --topology-rowblocks-neighbor-policy "per_neighbor"
    echo ""

    # 4) DP + chunk (hybrid) — physical batch 12 for GPU memory, logical batch BATCH_SIZE
    echo "[4/4] DP + chunk (hybrid)  er_p=$er_p  seed=$seed"
    python3 -u main.py "${BASE_ARGS[@]}" --batch-size 12 --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed" --dp --dp-noise "$DP_NOISE" --chunk --chunking-mode "topology_rowblocks" --topology-rowblocks-neighbor-policy "per_neighbor"
    echo ""
  done
done

echo "=== Done. CSVs in $RESULTS_ROOT/er_p_<p> (one file per condition per seed per er_p). ==="
echo "Plot per er_p (repeat for each p in ER_PS; add --additional-results-dir for each fixed-K sweep):"
echo "  python3 -m plotting.hybrid_ablation.hybrid_privacy_tradeoff \\"
echo "    --results-dir $RESULTS_ROOT/er_p_<p> \\"
echo "    --additional-results-dir results/fixed_k_chunking_sweep/8_standard_chunking_sweep/er_p_<p> \\"
echo "    --additional-results-dir results/fixed_k_chunking_sweep/16_standard_chunking_sweep/er_p_<p> \\"
echo "    --additional-results-dir results/fixed_k_chunking_sweep/32_standard_chunking_sweep/er_p_<p> \\"
echo "    --additional-results-dir results/fixed_k_chunking_sweep/64_standard_chunking_sweep/er_p_<p> \\"
echo "    --additional-results-dir results/fixed_k_chunking_sweep/128_standard_chunking_sweep/er_p_<p> \\"
echo "    --out-dir plots/hybrid_ablation/er_p_<p> --lambda 0.5 --auc-col max_auc"