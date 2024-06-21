#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <CONFIG>"
    exit 1
fi

CONFIG=$1
PORT=6379

if ! nc -z localhost $PORT; then
    echo "Starting Ray on port $PORT"
    ray start --head --port=$PORT --dashboard-port=0
else
    echo "Ray is already running on port $PORT"
fi

OUTNAME=$(echo "$CONFIG" | sed -n 's/.*\/\([^/]*\)\.json/\1/p')
python q2_ritme/run_n_eval_tune.py --config "$CONFIG" > x_"$OUTNAME"_out.txt 2>&1
