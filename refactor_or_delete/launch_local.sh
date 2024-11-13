#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <CONFIG>"
    exit 1
fi

CONFIG=$1

OUTNAME=$(echo "$CONFIG" | sed -n 's/.*\/\([^/]*\)\.json/\1/p')
python q2_ritme/run_n_eval_tune.py --config "$CONFIG" > x_"$OUTNAME"_out.txt 2>&1
