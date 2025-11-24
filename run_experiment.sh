#!/bin/bash

cleanup() {
    echo "Killing all spawned processes..."
    PGID=$(ps -o pgid= $$ | grep -o '[0-9]*')
    kill -TERM -"$PGID"
}

trap cleanup SIGINT

python3 main.py --rounds 10 --peers 10 --seed 123 \
 --topology "ring" --model "cnn" --dataset "cifar10" \
 --eval --eval-interval 3  --time-rounds &
# > /dev/null 2>&1
echo "Waiting for experiment to complete..."
wait

echo "Experiment done"