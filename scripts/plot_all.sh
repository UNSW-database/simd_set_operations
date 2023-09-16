#!/usr/bin/env sh

python3 scripts/plot.py $1 &&
python3 scripts/summary.py $1 > plots/index.html &&
echo plots/index.html
