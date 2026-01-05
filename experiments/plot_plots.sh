#!/bin/bash
cd .. # Move to root directory (where main.py lives)
cd src
cd plotting
cleanup() {
    echo "Killing all spawned processes..."
    PGID=$(ps -o pgid= $$ | grep -o '[0-9]*')
    kill -TERM -"$PGID"
}

trap cleanup SIGINT

python3 alpha_vs_auc.py --results-glob "../../results/allnodes/*message*.csv" \
    --output-dir "../../plots/alpha_sweep/allnodes" &

# > /dev/null 2>&1
echo "Waiting for experiment to complete..."
wait

echo "Experiment done"