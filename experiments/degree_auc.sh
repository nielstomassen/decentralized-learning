#!/bin/bash
cd .. # Move to root directory (where main.py lives)
cleanup() {
    echo "Killing all spawned processes..."
    PGID=$(ps -o pgid= $$ | grep -o '[0-9]*')
    kill -TERM -"$PGID"
}

trap cleanup SIGINT

ALPHAS=(0.1 0.3 0.6 1)
DEGREES=(1 2 3 4 5 6 7 8 9)
SEEDS=(123 555 4242)

MESSAGE_TYPE="delta"
TOPOLOGY="regular"
MODEL="cnn"
DATASET="mnist"
ROUNDS=30
PEERS=10
LOCAL_EPOCHS=1
MIA_ATTACK="baseline"
MIA_INTERVAL=30
MIA_TYPE="loss"
PARTITIONER="dirichlet"
MIA_RESULTS_ROOT="results/degree"

mkdir -p logs

echo "Running alpha sweep:  ${ALPHAS[@]}"
echo "Using degrees:        ${DEGREES[@]}"
echo "Using seeds:          ${SEEDS[@]}"
echo
# ============================================
# Launch experiments in parallel
# ============================================

MAX_JOBS=1
job_count=0
for ALPHA in "${ALPHAS[@]}"; do
    for DEGREE in "${DEGREES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
    
        LOGFILE="logs/deg/alpha_${ALPHA}_degree_${DEGREE}_seed_${SEED}.log"
        echo "Starting run for alpha=${ALPHA}, degree=${DEGREE}, seed=${SEED}"

        python3 main.py \
        --rounds $ROUNDS \
        --peers $PEERS \
        --seed $SEED \
        --topology "$TOPOLOGY" \
        --regular-degree $DEGREE \
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
        --mia-results-root $MIA_RESULTS_ROOT \
        > "$LOGFILE" 2>&1 &

        ((job_count++))

        if (( job_count >= MAX_JOBS )); then
        # Wait for any one job to finish (bash 5+)
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
