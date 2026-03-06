#!/bin/bash
# Sweep number of chunks (d = degree) with chunking on and chunks_per_neighbor=1.
# Use random_regular topology so every node has exactly DEGREE neighbors = DEGREE chunks.
# Find optimal d (e.g. 10) then use --regular-degree 10 or ER p that gives ~10 neighbors.

set -e
cd "$(dirname "$0")/.."

ROUNDS=${ROUNDS:-60}
PEERS=${PEERS:-100}
SEEDS="${SEEDS:-42 123 456}"
# Degrees = number of chunks per node (each neighbor gets 1 chunk)
DEGREE_LIST="${DEGREE_LIST:-2 4 6 8 10 12 16 20}"
RESULTS_ROOT=${RESULTS_ROOT:-results/chunk_degree_sweep}

BASE_ARGS=(
  --rounds "$ROUNDS"
  --peers "$PEERS"
  --topology "random_regular"
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
  --chunks-per-neighbor 1
)

echo "=== Chunk degree sweep: number of chunks (d=degree) over $DEGREE_LIST ==="
echo "  RESULTS_ROOT=$RESULTS_ROOT  ROUNDS=$ROUNDS  PEERS=$PEERS"
echo "  SEEDS=$SEEDS"
echo ""

for deg in $DEGREE_LIST; do
  OUT_DIR="${RESULTS_ROOT}/deg_${deg}"
  mkdir -p "$OUT_DIR"
  for seed in $SEEDS; do
    echo "[degree=$deg seed=$seed]"
    python3 main.py "${BASE_ARGS[@]}" \
      --regular-degree "$deg" \
      --mia-results-root "$OUT_DIR" \
      --seed "$seed"
    echo ""
  done
done

echo "=== Done. CSVs in $RESULTS_ROOT ==="
echo "Summarize: python3 experiments/summarize_chunk_sweep.py --results-dir $RESULTS_ROOT --by degree"
