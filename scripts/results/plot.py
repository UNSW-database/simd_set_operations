#!/usr/bin/env python3
import sys
import os
import pandas as pd
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
    "throughput/density": "Throughput / density",
    "time_s": "Intersection Time (s)",
    "time_s/element": "Time per Element (s)",
    "time_ns": "Intersection Time (ns)",
    "time_ns/element": "Time per Element (ns)",
    "branches": "Total Branches",
    "branches/element": "Branches per Element",
    "branch_misses": "Total Branch Misses",
    "branch_miss_rate": "Branch Miss Rate",
    "branch_misses/element": "Branch Misses per Element",
    "instructions": "Total Retired Instructions",
    "instructions/element": "Retired Instructions per Element",
    "cpi": "Cycles per Instruction",
    "ipc": "Instructions per Clock",
    "cpu_cycles": "Total CPU Cycles",
    "cpu_cycles/element": "CPU Cycles per Element",
    "cpu_cycles_ref": "Total CPU Cycles (adjusted for frequency scaling)",
    "cpu_cycles_ref/element": "CPU Cycles per Element (adjusted for frequency scaling)",
}

SCALAR_ALGORITHMS = {
    "naive_merge": "Merge (branch)",
    "branchless_merge": "Merge (branchless)",
}

VECTOR_ALGORITHMS = {
    "qfilter": "QFilter",
    "qfilter_c": "QFilter (FFI)",
    "bmiss": "BMiss",
    "bmiss_sttni": "BMiss STTNI",
    "shuffling_sse": "Shuffling SSE",
    "shuffling_avx2": "Shuffling AVX2",
    "shuffling_avx512": "Shuffling AVX512",
    "broadcast_sse": "Broadcast SSE",
    "broadcast_avx2": "Broadcast AVX2",
    "broadcast_avx512": "Broadcast AVX512",
    "vp2intersect_emulation": "VP2INT. Emul.",
    "croaring": "C Roaring",
    "lbk_v3_sse": "LBK v3 SSE",
    "lbk_v3_avx2": "LBK v3 AVX2",
    "lbk_v3_avx512": "LBK v3 AVX512",
    "lbk_v1x4_sse": "LBK v1 x4 SSE",
    "lbk_v1x8_sse": "LBK v1 x8 SSE",
    "lbk_v1x8_avx2": "LBK v1 x8 AVX2",
    "lbk_v1x16_avx2": "LBK v1 x16 AVX2",
    "lbk_v1x16_avx512": "LBK v1 x16 AVX512",
    "lbk_v1x32_avx512": "LBK v1 x32 AVX512",
    "galloping": "Galloping",
    "galloping_sse": "Galloping SSE",
    "galloping_avx2": "Galloping AVX2",
    "galloping_avx512": "Galloping AVX512",
}

SAVE_TYPE = {
    "_count": "count",
    "_lut": "lookup",
    "_comp": "compress",
}

NO_SAVE = [
    "merge",
    "bmiss",
    "galloping",
    "lbk",
    "croaring",
]

NO_BRANCH = [
    "merge",
    "galloping",
    "lbk",
    "croaring",
]

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
        fig = plot_y_vs_x(all_results, x_col, y_col)

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

def plot_y_vs_x(all_results, x_col, y_col):
    fig, ax = plt.subplots()
    ax.set_xlabel(col_title(x_col))
    ax.set_ylabel(col_title(y_col))
    # ax.set_title("Relative throughput")
    ax.grid(True)

    if do_log(x_col):
        ax.set_xscale("log")
    if do_log(y_col):
        ax.set_yscale("log")

    x_formatter = col_formatter(x_col)
    if x_formatter is not None:
        ax.xaxis.set_major_formatter(lambda x, _: x_formatter(x))

    for alg, df in all_results.items():
        ax.plot(df[x_col], df[y_col], label=algorithm_label(alg))

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
        alg_names = [algorithm_label(alg) for alg in algorithms]
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

        ax.set_xlabel(col_title(col))
        if i % MAX_COLS == 0:
            ax.set_ylabel("Algorithm")
        
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
    
    label_row_indices = [int(x) for x in labels]
    labels = [f"Selectivity {point_results[i][0]}, Density {point_results[i][1]}, Skew 1:{point_results[i][2]:.0f}, Size {point_results[i][3]}" for i in label_row_indices]
    fig.legend(handles, labels, loc="lower center", ncol=3)

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
            return f"{CACHE_NAMES[cache.group(1)]} {CACHE_OP[cache.group(2)]} {CACHE_EVENT[cache.group(3)]}{ ' per element' if cache.group(4) else ''}"

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
        return None

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

def algorithm_label(alg, label="", has_branch=False):

    scalar = SCALAR_ALGORITHMS.get(alg)
    if scalar is not None:
        return scalar + label

    vector = VECTOR_ALGORITHMS.get(alg)
    if vector is not None:
        if not has_branch and not any(x in alg for x in NO_BRANCH):
            label = " (branchless)" + label
        return vector + label
    
    for postfix, tag in SAVE_TYPE.items():
        if alg.endswith(postfix):
            rest = alg[:-len(postfix)]
            tagged = f" ({tag})" + label if not any(x in alg for x in NO_SAVE) else label
            return algorithm_label(rest, tagged, has_branch) 
    
    if alg.endswith("_br"):
        return algorithm_label(alg[:-3], " (branch)" + label, True) 
    
    if alg.endswith("_bsr"):
        if not has_branch and not any(x in alg for x in NO_BRANCH):
            label = " (branchless)" + label
        return algorithm_label(alg[:-4], " BSR" + label, True)
            
    return f"FAIL: {alg}{label}"

if __name__ == "__main__":
    main()
