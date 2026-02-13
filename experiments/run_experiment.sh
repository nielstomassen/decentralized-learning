#!/bin/bash
cd .. # Move to root directory (where main.py lives)
cleanup() {
    echo "Killing all spawned processes..."
    PGID=$(ps -o pgid= $$ | grep -o '[0-9]*')
    kill -TERM -"$PGID"
}

trap cleanup SIGINT

python3 main.py --rounds 150 --peers 20 --seed 4 \
 --topology "star" --er-p 0.16 --beta 0.7 --model "cnn" --dataset "cifar100" \
 --time-rounds --eval --local-epochs 1 --mia-attack "baseline" --mia-interval 50 \
 --batch-size 16 --mia-baseline-type "loss" --partitioner "dirichlet" --alpha 10 \
 --weight-decay 0 --dp-noise 0.5 --mia-results-root "results/cifar100/star/dp" \
 --message-type "full" --dp --mia-measurement-number 1000 --no-samples 500 &

# > /dev/null 2>&1
echo "Waiting for experiment to complete..."
wait

echo "Experiment done"