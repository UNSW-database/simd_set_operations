#!/usr/bin/env bash

HTML=${2-'plots/index.html'}

python3 scripts/plot.py $1 $3 &&
python3 scripts/summary.py $1 $3 > "$HTML" &&
echo "$HTML"
