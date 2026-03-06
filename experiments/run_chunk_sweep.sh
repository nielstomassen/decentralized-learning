#!/bin/bash
# Sweep chunks_per_neighbor (with chunking on) to find optimal utility vs AUC.
# Each run uses the same topology; only --chunks-per-neighbor varies.
# Results go to RESULTS_ROOT; use summarize_chunk_sweep.py to get a table of utility & AUC per cpn.

set -e
cd "$(dirname "$0")/.."

ROUNDS=${ROUNDS:-60}
PEERS=${PEERS:-100}
SEEDS="${SEEDS:-42 123 456}"
# Chunks per neighbor values to sweep (each neighbor receives this many of the d chunks)
CHUNKS_PER_NEIGHBOR_LIST="${CHUNKS_PER_NEIGHBOR_LIST:-1 2 3 4 5 6 8 10}"
# Topology: er with fixed p (fixes average degree ≈ number of chunks d)
ER_P=${ER_P:-0.08}
RESULTS_ROOT=${RESULTS_ROOT:-results/chunk_sweep}

BASE_ARGS=(
  --rounds "$ROUNDS"
  --peers "$PEERS"
  --topology "er"
  --er-p "$ER_P"
  --beta 0.7
  --model "cnn"
  --dataset "cifar100"
  --local-epochs 5
  --mia-attack "baseline"
  --mia-interval 60
  --batch-size 16
  --mia-baseline-type "loss"
  --partitioner "iid"
  --alpha 10
  --weight-decay 0
  --message-type "full"
  --mia-measurement-number 1000
  --no-samples 500
  --eval-top-k 5
  --eval
  --chunk
)

echo "=== Chunk sweep: chunks_per_neighbor over $CHUNKS_PER_NEIGHBOR_LIST (chunking on) ==="
echo "  RESULTS_ROOT=$RESULTS_ROOT  ROUNDS=$ROUNDS  PEERS=$PEERS  ER_P=$ER_P"
echo "  SEEDS=$SEEDS"
echo ""

mkdir -p "$RESULTS_ROOT"

for cpn in $CHUNKS_PER_NEIGHBOR_LIST; do
  for seed in $SEEDS; do
    echo "[cpn=$cpn seed=$seed]"
    python3 main.py "${BASE_ARGS[@]}" \
      --chunks-per-neighbor "$cpn" \
      --mia-results-root "$RESULTS_ROOT" \
      --seed "$seed"
    echo ""
  done
done

echo "=== Done. CSVs in $RESULTS_ROOT ==="
echo "Summarize: python3 experiments/summarize_chunk_sweep.py --results-dir $RESULTS_ROOT"
