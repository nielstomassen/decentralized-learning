#!/bin/bash
# Appendix chunk variant: topology-aware chunking with broadcast_same (flat degree).
# Compare against normal hybrid ablation via normal_vs_broadcast_same_ta_hybrid.py

set -e
cd "$(dirname "$0")/.."
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/chunk_variant_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1
ROUNDS=${ROUNDS:-100}
PEERS=${PEERS:-100}
SEEDS="${SEEDS:-4235 45 929 9838 12}"
ER_PS="${ER_PS:-0.08 0.16}"
DP_NOISE=${DP_NOISE:-0.5}
RESULTS_ROOT=${RESULTS_ROOT:-results/appendix/flatBroadcastSame}
BATCH_SIZE=${BATCH_SIZE:-32}
CHUNK_ARGS=(
  --chunk
  --chunking-mode topology_flat_degree
  --topology-rowblocks-neighbor-policy broadcast_same
)
EXTRA_ARGS=()
[[ -n "${TIMING:-}" ]] && [[ "${TIMING}" != "0" ]] && EXTRA_ARGS+=(--time-rounds)
[[ -n "${DEVICE:-}" ]] && EXTRA_ARGS+=(--device "$DEVICE")

BASE_ARGS=(
  --rounds "$ROUNDS"
  --peers "$PEERS"
  --topology er
  --beta 0.5
  --model cnn
  --dataset cifar100
  --local-epochs 5
  --mia-attack baseline
  --mia-interval 100
  --batch-size "$BATCH_SIZE"
  --dp-logical-batch-size "$BATCH_SIZE"
  --mia-baseline-type loss
  --partitioner iid
  --alpha 0.3
  --weight-decay 0
  --message-type full
  --mia-measurement-number 1000
  --no-samples 1000
  --eval-top-k 5
  "${EXTRA_ARGS[@]}"
)

echo "=== Chunk variant (broadcast_same): topology-aware chunk + ChunkDP ==="
echo "  RESULTS_ROOT=$RESULTS_ROOT  ROUNDS=$ROUNDS  PEERS=$PEERS  DP_NOISE=$DP_NOISE  BATCH_SIZE=$BATCH_SIZE"
echo "  ER_PS=$ER_PS  SEEDS=$SEEDS  TIMING=${TIMING:-0}  DEVICE=${DEVICE:-auto}"
echo ""

for er_p in $ER_PS; do
  RESULTS_DIR="${RESULTS_ROOT}/er_p_${er_p}"
  mkdir -p "$RESULTS_DIR"
  echo "--- er_p=$er_p  RESULTS_DIR=$RESULTS_DIR ---"

  for seed in $SEEDS; do
    echo "[1/2] Topology-aware chunk (broadcast_same)  er_p=$er_p  seed=$seed"
    python3 -u main.py "${BASE_ARGS[@]}" \
      --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed" \
      "${CHUNK_ARGS[@]}"
    echo ""

    echo "[2/2] ChunkDP (broadcast_same)  er_p=$er_p  seed=$seed"
    python3 -u main.py "${BASE_ARGS[@]}" --batch-size 12 \
      --er-p "$er_p" --mia-results-root "$RESULTS_DIR" --seed "$seed" \
      --dp --dp-noise "$DP_NOISE" \
      "${CHUNK_ARGS[@]}"
    echo ""
  done
done

echo "=== Done. CSVs in $RESULTS_ROOT/er_p_<p>/ (topology-aware chunk + ChunkDP, one file per seed). ==="
echo "Plot vs normal hybrid ablation:"
echo "  python3 -m plotting.appendix.normal_vs_broadcast_same_ta_hybrid \\"
echo "    --factorial-root results/hybrid_ablation \\"
echo "    --broadcast-root $RESULTS_ROOT \\"
echo "    --out-path plots/appendix/chunk_variant/normal_vs_broadcast_same_ta_hybrid.png"
