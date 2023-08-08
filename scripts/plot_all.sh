#!/usr/bin/env sh

python3 scripts/plot.py &&
python3 scripts/summary.py > plots/index.html &&
echo plots/index.html
