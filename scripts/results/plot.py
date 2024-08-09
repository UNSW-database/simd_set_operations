#!/usr/bin/env python3
import sys
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import label

def main():
    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument("results_path", type=str, help="Path to results")
    parser.add_argument("out_path", type=str, help="Path to output file")
    parser.add_argument("--y_vs_x", action="store_true", help="Plot column y vs column x")
    parser.add_argument("--bars_per_alg", action="store_true", help="Plot bars for column per algorithm")
    parser.add_argument("--cols", type=str, nargs="+", help="Columns to plot")
    parser.add_argument("--xvalues", type=int, nargs="+", help="X values to plot")
    parser.add_argument("--without", type=str, nargs="*", help="Remove these algorithms")
    parser.add_argument("--only", type=str, nargs="*", help="Only use these algorithms")
    parser.add_argument("--log", action="store_true", help="Use log scale")
    args = parser.parse_args()

    results_path = Path(args.results_path)
    out_path = Path(args.out_path)

    all_results = {}
    for file in os.listdir(results_path):
        all_results[Path(file).stem] = pd.read_csv(results_path / file)

    if args.only:
        all_results = {k: v for k, v in all_results.items() if k in args.only}
    if args.without:
        all_results = {k: v for k, v in all_results.items() if k not in args.without}

    if args.y_vs_x:
        args.cols is not None and len(args.cols) == 2 or fail("Expected two columns for y_vs_x")
        y_col, x_col = args.cols
        fig = plot_y_vs_x(all_results, x_col, y_col, args.log)

    elif args.bars_per_alg:
        args.xvalues is not None and len(args.xvalues) >= 1 or fail("Expected at least one argument for xvalues")
        args.cols is not None and len(args.cols) >= 1 or fail("Expected at least one argument for cols")
        fig = plot_bars_per_algorithm(all_results, args.cols, args.xvalues)

    else:
        fail("No plot type specified")

    fig.savefig(out_path)
    print(f"{out_path}")

def fail(msg):
    print(msg)
    sys.exit(1)

def plot_y_vs_x(all_results, x_col, y_col, use_log):
    fig, ax = plt.subplots()
    ax.set_xlabel(label.col_title(x_col))
    ax.set_ylabel(label.col_title(y_col))
    # ax.set_title("Relative throughput")
    ax.grid(True)

    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")

    if label.do_log(x_col):
        ax.set_xscale("log")
    if label.do_log(y_col):
        ax.set_yscale("log")

    x_formatter = label.col_formatter(x_col)
    if x_formatter is not None:
        ax.xaxis.set_major_formatter(lambda x, _: x_formatter(x))

    for alg, df in all_results.items():
        ax.plot(df[x_col], df[y_col], label=label.algorithm_label(alg))

    ax.legend()
    return fig

def plot_bars_per_algorithm(all_results, columns, row_indices):

    def get_point_results(row_idx):
        some_alg = next(iter(all_results.keys()))
        sel = all_results[some_alg]["selectivity"][row_idx]
        density = all_results[some_alg]["density"][row_idx]
        skew_f = all_results[some_alg]["skewness_factor"][row_idx]
        size = all_results[some_alg]["max_len"][row_idx]
        skew = 2 ** skew_f
        return sel, density, skew, size
    
    point_results = {row_idx: get_point_results(row_idx) for row_idx in row_indices}

    algorithms = list(all_results.keys())

    dfs = {}
    for col in columns:
        alg_names = [label.algorithm_label(alg) for alg in algorithms]
        df = pd.DataFrame(index=alg_names)
        for index in row_indices:
            df[index] = [all_results[alg][col][index] for alg in algorithms]
        dfs[col] = df

    MAX_COLS = 3
    ncols = min(MAX_COLS, len(columns))
    nrows = (len(columns) + MAX_COLS - 1) // MAX_COLS
    row_height = 0.5 * len(algorithms)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(5 * ncols, row_height * nrows))
    fig.subplots_adjust(left=0.18, wspace=0.15, hspace=3 / len(algorithms), bottom=0.15, top=0.95, right=0.98)

    fig.suptitle(f"CPU statistics per algorithm")

    for i, col in enumerate(columns):
        ax = axs[i // MAX_COLS, i % MAX_COLS]

        dfs[col].plot(ax=ax, kind="barh", stacked=True, legend=False)

        ax.set_xlabel(label.col_title(col))
        if i % MAX_COLS == 0:
            ax.set_ylabel("Algorithm")
        
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
    
    label_row_indices = [int(x) for x in labels]
    labels = [f"Selectivity {point_results[i][0]}, Density {point_results[i][1]}, Skew 1:{point_results[i][2]:.0f}, Size {point_results[i][3]}" for i in label_row_indices]
    fig.legend(handles, labels, loc="lower center", ncol=3)

    return fig

if __name__ == "__main__":
    main()
