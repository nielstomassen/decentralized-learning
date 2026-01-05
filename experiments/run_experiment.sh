#!/bin/bash
cd .. # Move to root directory (where main.py lives)
cleanup() {
    echo "Killing all spawned processes..."
    PGID=$(ps -o pgid= $$ | grep -o '[0-9]*')
    kill -TERM -"$PGID"
}

trap cleanup SIGINT

python3 main.py --rounds 1 --peers 6 --seed 444 \
 --topology "regular" --regular-degree 5 --model "cnn" --dataset "mnist" \
 --time-rounds --local-epochs 1 --mia-attack "baseline" --mia-interval 1 \
 --mia-baseline-type "loss" --partitioner "dirichlet" --alpha 0.1 \
 --mia-results-root "results/test" &

# > /dev/null 2>&1
echo "Waiting for experiment to complete..."
wait

echo "Experiment done"