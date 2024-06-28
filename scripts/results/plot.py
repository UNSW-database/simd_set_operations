#!/usr/bin/env python3
import sys
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import re
import argparse

CACHE_NAMES = {
    "l1d": "L1 Data",
    "l1i": "L1 Instr.",
    "ll": "Last Level",
}
CACHE_OP = {
    "rd": "Read",
    "wr": "Write",
}
CACHE_EVENT = {
    "access": "Accesses",
    "miss": "Misses",
    "miss_rate": "Miss rate",
}

COL_TITLES = {
    "selectivity": "Selectivity",
    "density": "Density",
    "size": "Size",
    "skewness_factor": "Skew",
    "set_count": "Set Count",
    "element_count": "Element Count",
    "element_bytes": "Total size of all sets (bytes)",
    "element_bytes_pow": "Total size of all sets (bytes)",
    "throughput_eps": "Throughput (elements/s)",
    "time_s": "Intersection Time (s)",
    "time_ns": "Intersection Time (ns)",
    "branches": "Total Branches",
    "branch_misses": "Total Branch Misses",
    "branch_miss_rate": "Branch Miss Rate",
    "cpu_cycles": "Total CPU Cycles",
}

def main():
    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument("results_path", type=str, help="Path to results")
    parser.add_argument("out_path", type=str, help="Path to output file")
    parser.add_argument("--y_vs_x", type=str, nargs=2, help="Plot column y vs column x")
    parser.add_argument("--bars_per_alg", type=str, nargs="+", help="Plot bars for column per algorithm")
    args = parser.parse_args()

    results_path = Path(args.results_path)
    out_path = Path(args.out_path)

    all_results = {}
    for file in os.listdir(results_path):
        all_results[Path(file).stem] = pd.read_csv(results_path / file)

    if args.y_vs_x is not None:
        y_col, x_col = args.y_vs_x
        fig = plot_y_vs_x(all_results, x_col, y_col)
    elif args.bars_per_alg is not None:
        if len(args.bars_per_alg) < 2:
            fail("Expected at least two arguments for bars_per_alg")
        row_idx = int(args.bars_per_alg[0])
        columns = args.bars_per_alg[1:]
        fig = plot_bars_per_algorithm(all_results, columns, row_idx)
    else:
        fail("No plot type specified")

    fig.savefig(out_path)
    print(f"{out_path}")

def fail(msg):
    print(msg)
    sys.exit(1)

def plot_y_vs_x(all_results, x_col, y_col):
    fig, ax = plt.subplots()
    ax.set_xlabel(col_title(x_col))
    ax.set_ylabel(col_title(y_col))
    ax.set_title("Relative throughput")
    ax.grid(True)

    if do_log(x_col):
        ax.set_xscale("log")
    if do_log(y_col):
        ax.set_yscale("log")

    x_formatter = col_formatter(x_col)
    ax.xaxis.set_major_formatter(lambda x, _: x_formatter(x))

    for alg, df in all_results.items():
        ax.plot(df[x_col], df[y_col], label=alg)

    ax.legend()
    return fig

def plot_bars_per_algorithm(all_results, columns, row_idx):

    values = pd.DataFrame()
    values["algorithm"] = list(all_results.keys())

    some_alg = next(iter(all_results.keys()))
    sel = all_results[some_alg]["selectivity"][row_idx]
    density = all_results[some_alg]["density"][row_idx]
    skew_f = all_results[some_alg]["skewness_factor"][row_idx]
    size = all_results[some_alg]["max_len"][row_idx]

    for col in columns:
        values[col] = [all_results[alg][col][row_idx] for alg in values["algorithm"]]
    
    MAX_COLS = 3
    ncols = min(MAX_COLS, len(columns))
    nrows = (len(columns) + MAX_COLS - 1) // MAX_COLS
    row_height = 0.7 * len(all_results.keys())

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(5 * ncols, row_height * nrows))
    fig.subplots_adjust(left=0.18, wspace=0.15, hspace=0.5, bottom=0.5, top=0.95, right=0.98)

    skew = 2 ** skew_f
    fig.suptitle(f"CPU statistics per algorithm (sets of size {size}, skew 1:{skew}, selectivity {sel}, density {density})")

    for i, col in enumerate(columns):
        ax = axs[i // MAX_COLS, i % MAX_COLS]

        ax.barh(values["algorithm"], values[col])

        ax.set_xlabel(col_title(col))
        if i % MAX_COLS == 0:
            ax.set_ylabel("Algorithm")

    return fig

def col_title(col): 
    title = COL_TITLES.get(col)
    if title is not None:
        return title
    else:
        relative = re.match(r"^throughput_vs_(.*)$", col)
        if relative:
            return f"Relative throughput ({relative.group(1)}=1)"
        
        cache = re.match(r"^(l1d|l1i|ll)_(rd|wr)_(access|miss|miss_rate)(/element)?$", col)
        if cache:
            return f"{CACHE_NAMES[cache.group(1)]} {CACHE_OP[cache.group(2)]} {CACHE_EVENT[cache.group(3)]}{ '/ element' if cache.group(4) else ''}"

        print(f"Unknown col: {col}")
        sys.exit(1)

def col_formatter(col):
    if col == "element_bytes":
        return lambda x: format_unit(x, 1000, ['B', 'KB', 'MB', 'GB', 'TB'])
    if col == "element_bytes_pow":
        return lambda x: format_unit_pow(x, 10, ['B', 'KiB', 'MiB', 'GiB', 'TiB'])
    elif col == "time_ns":
        return lambda x: format_unit(x, 1000, ['ns', 'us', 'ms', 's'])
    else:
        return lambda x: x

def format_unit(value, power, labels):
    n = 0
    while value > power and n < len(labels) - 1:
        value /= power
        n += 1
    return "{:.0f}".format(value) + labels[n]

def format_unit_pow(value, exp, labels):
    unit = int(value) // exp
    offset = int(value) % exp
    return f"{1 << offset}{labels[unit]}"

def do_log(col):
    return col in ["element_count", "element_bytes"]

if __name__ == "__main__":
    main()
