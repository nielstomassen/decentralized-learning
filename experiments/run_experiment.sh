#!/bin/bash
cd .. # Move to root directory (where main.py lives)
cleanup() {
    echo "Killing all spawned processes..."
    PGID=$(ps -o pgid= $$ | grep -o '[0-9]*')
    kill -TERM -"$PGID"
}

trap cleanup SIGINT

python3 main.py --rounds 5 --peers 100 --seed 4 \
 --topology "er" --er-p 0.04 --beta 0.7 --model "cnn" --dataset "cifar10" \
 --time-rounds --eval --local-epochs 2 --mia-attack "baseline" --mia-interval 5 \
 --batch-size 64 --mia-baseline-type "loss" --partitioner "iid" --alpha 0.3 \
 --mia-results-root "results/new/60/1epoch" --message-type "full" --mia-measurement-number 1000 &

# > /dev/null 2>&1
echo "Waiting for experiment to complete..."
wait

echo "Experiment done"