#!/bin/bash
cd .. # Move to root directory (where main.py lives)
cleanup() {
    echo "Killing all spawned processes..."
    PGID=$(ps -o pgid= $$ | grep -o '[0-9]*')
    kill -TERM -"$PGID"
}

trap cleanup SIGINT

SEEDS=(555 42 91 6060 49 1 406 5555 82828 683 258 2354 30636 693 14827)
ER_PS=(0.08)

TOPOLOGY="er"
ALPHA=0.8
MODEL="cnn"
DATASET="cifar10"
BETA=0.8
ROUNDS=100
PEERS=100
LOCAL_EPOCHS=5
MIA_ATTACK="baseline"
MIA_INTERVAL=100
MIA_N=1000
MIA_TYPE="loss"
PARTITIONER="iid"
MESSAGE_TYPE="full"
MIA_RESULTS_ROOT="results/highbeta/nochunk"

mkdir -p logs

echo "Running er_p:  ${ER_PS[@]}"
echo "Using seeds:          ${SEEDS[@]}"
echo
# ============================================
# Launch experiments in parallel
# ============================================

MAX_JOBS=1
job_count=0
for ER_P in "${ER_PS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

    LOGFILE="logs/ring/er_${ER_P}_topology_${TOPOLOGY}_seed_${SEED}.log"
    echo "Starting run for p=${ER_P}, seed=${SEED}"

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
    --mia-results-root $MIA_RESULTS_ROOT \
    --beta $BETA \
    > "$LOGFILE" 2>&1 &

    ((job_count++))

    if (( job_count >= MAX_JOBS )); then
    # Wait for any one job to finish (bash 5+)
    wait -n
    ((job_count--))
    fi
        
  done
done

echo
echo "Waiting for all experiments to complete..."
wait

echo "All experiments finished."
