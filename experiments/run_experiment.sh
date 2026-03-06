#!/bin/bash
cd .. # Move to root directory (where main.py lives)
cleanup() {
    echo "Killing all spawned processes..."
    PGID=$(ps -o pgid= $$ | grep -o '[0-9]*')
    kill -TERM -"$PGID"
}

trap cleanup SIGINT

python3 main.py --rounds 30 --peers 100 --seed 4 \
 --topology "er" --er-p 0.04 --beta 0.7 --model "cnn" --dataset "cifar100" \
 --time-rounds --eval --local-epochs 5 --mia-attack "baseline" --mia-interval 60 \
 --batch-size 16 --mia-baseline-type "loss" --partitioner "dirichlet" --alpha 10 \
 --weight-decay 0 --dp-noise 0.5 --mia-results-root "results/delta" --one-attacker \
 --message-type "full" --mia-measurement-number 1000 --no-samples 500 --eval-top-k 5 &

# > /dev/null 2>&1
echo "Waiting for experiment to complete..."
wait

echo "Experiment done"