#!/bin/bash
# Peer-count sweep: baseline vs topology-aware chunking peer count (appendix experiment)
# Loops ER_PS (default 0.08 0.16) and PEER_COUNTS (default 10 25 50 75); fixed ER topology only.

set -e
cd "$(dirname "$0")/.."
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/peer_count_mechanism_sweep_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1
ROUNDS=${ROUNDS:-100}
PEER_COUNTS="${PEER_COUNTS:-10 25 50 75}"
SEEDS="${SEEDS:-4235 45 929 9838 12}"
ER_PS="${ER_PS:-0.08 0.16}"
TOPOLOGY=${TOPOLOGY:-er}
DP_NOISE=${DP_NOISE:-0.5}
RESULTS_ROOT=${RESULTS_ROOT:-results/peer_count_mechanism_sweep}
BATCH_SIZE=${BATCH_SIZE:-32}
EXTRA_ARGS=()
[[ -n "${TIMING:-}" ]] && [[ "${TIMING}" != "0" ]] && EXTRA_ARGS+=(--time-rounds)
[[ -n "${DEVICE:-}" ]] && EXTRA_ARGS+=(--device "$DEVICE")

echo "=== Peer-count mechanism sweep: 2×2 (DP × chunk) ==="
echo "  RESULTS_ROOT=$RESULTS_ROOT  ROUNDS=$ROUNDS  PEER_COUNTS=$PEER_COUNTS"
echo "  TOPOLOGY=$TOPOLOGY  ER_PS=$ER_PS  DP_NOISE=$DP_NOISE  BATCH_SIZE=$BATCH_SIZE"
echo "  SEEDS=$SEEDS  TIMING=${TIMING:-0}  DEVICE=${DEVICE:-auto}"
echo ""

for er_p in $ER_PS; do
  for peers in $PEER_COUNTS; do
    RESULTS_DIR="${RESULTS_ROOT}/er_p_${er_p}/peers_${peers}"
    mkdir -p "$RESULTS_DIR"
    echo "--- er_p=$er_p  peers=$peers  topology=$TOPOLOGY  RESULTS_DIR=$RESULTS_DIR ---"

    BASE_ARGS=(
      --rounds "$ROUNDS"
      --peers "$peers"
      --topology "$TOPOLOGY"
      --er-p "$er_p"
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
    --mia-measurement-number 400
    --no-samples 400
    --eval-top-k 5
    "${EXTRA_ARGS[@]}"
  )

    for seed in $SEEDS; do
      echo "[1/2] No DP, no chunk (baseline)  er_p=$er_p  peers=$peers  seed=$seed"
      python3 -u main.py "${BASE_ARGS[@]}" --mia-results-root "$RESULTS_DIR" --seed "$seed"
      echo ""

      echo "[2/2] Topology-aware chunk only  er_p=$er_p  peers=$peers  seed=$seed"
      python3 -u main.py "${BASE_ARGS[@]}" --mia-results-root "$RESULTS_DIR" --seed "$seed" --chunk \
        --chunking-mode topology_rowblocks --topology-rowblocks-neighbor-policy per_neighbor
      echo ""


    done
  done
done

echo "=== Done. CSVs in $RESULTS_ROOT/er_p_<p>/peers_<n>/ (2 conditions × seeds per er_p × peer count). ==="
echo "Plot: python3 src/plotting/hybrid_privacy_tradeoff.py --results-dir $RESULTS_ROOT/er_p_<p>/peers_<n> --out-dir plots/peer_count/er_p_<p>/peers_<n>"
