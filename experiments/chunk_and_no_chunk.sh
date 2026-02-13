#!/bin/bash
cd .. # Move to root directory (where main.py lives)

cleanup() {
    echo "Killing all spawned processes..."
    PGID=$(ps -o pgid= $$ | grep -o '[0-9]*')
    kill -TERM -"$PGID"
}

trap cleanup SIGINT

SEEDS=(555 42 91 6060 49 1 406 5555 82828 683 258 2354 30636 693 14827)
ER_PS=(0.04)

TOPOLOGY="er"
ALPHA=1.8
MODEL="cnn"
DATASET="emnist"
BETA=0.7
ROUNDS=30
PEERS=100
LOCAL_EPOCHS=2
MIA_ATTACK="baseline"
MIA_INTERVAL=30
MIA_N=1000
MIA_TYPE="loss"
PARTITIONER="dirichlet"
MESSAGE_TYPE="full"
WEIGHT_DECAY=0
NO_SAMPLES=500

# Two experiments per seed: one without chunking, one with chunking
CHUNK_MODES=("nochunk" "chunk")

mkdir -p logs

echo "Running er_p:         ${ER_PS[@]}"
echo "Using seeds:          ${SEEDS[@]}"
echo "Chunk modes:          ${CHUNK_MODES[@]}"
echo

# ============================================
# Launch experiments in parallel
# ============================================

MAX_JOBS=1
job_count=0

for ER_P in "${ER_PS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for CHUNK_MODE in "${CHUNK_MODES[@]}"; do

      # Put results into separate folders
      MIA_RESULTS_ROOT="results/emnist2/${CHUNK_MODE}"

      # Separate log per run
      LOGFILE="logs/ring/er_${ER_P}_topology_${TOPOLOGY}_seed_${SEED}_${CHUNK_MODE}.log"

      echo "Starting run for p=${ER_P}, seed=${SEED}, chunk_mode=${CHUNK_MODE}"

      # If chunking is enabled in your code via a flag, set it here.
      # Adjust the flag name/value to match your argparse options.
      CHUNK_FLAG=""
      if [[ "$CHUNK_MODE" == "chunk" ]]; then
        CHUNK_FLAG="--chunk"
      fi

      python3 main.py \
        --rounds $ROUNDS \
        --peers $PEERS \
        --seed $SEED \
        --topology "$TOPOLOGY" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --local-epochs $LOCAL_EPOCHS \
        --time-rounds \
        --mia-attack "$MIA_ATTACK" \
        --mia-interval $MIA_INTERVAL \
        --mia-baseline-type "$MIA_TYPE" \
        --partitioner "$PARTITIONER" \
        --message-type "$MESSAGE_TYPE" \
        --alpha $ALPHA \
        --no-samples $MIA_N \
        --er-p $ER_P \
        --mia-results-root "$MIA_RESULTS_ROOT" \
        --beta $BETA \
        --weight-decay $WEIGHT_DECAY \
        --no-samples $NO_SAMPLES \
        $CHUNK_FLAG \
        > "$LOGFILE" 2>&1 &

      ((job_count++))

      if (( job_count >= MAX_JOBS )); then
        wait -n
        ((job_count--))
      fi

    done
  done
done

echo
echo "Waiting for all experiments to complete..."
wait

echo "All experiments finished."
