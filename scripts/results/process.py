#!/usr/bin/env python3
import sys
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

DEFAULT_IN_PATH = "results/paper/tods-size-sse.json"
DEFAULT_OUT_PATH = "processed"
ELEMENT_BYTES = 4

def add_cpu_stat(df, alg_results, name, getter):
    avgs = []
    stds = []
        
    for row in alg_results:
        samples = getter(row)
        if samples is None:
            return df
        avgs.append(sum(samples) / len(samples))
        stds.append(np.std(samples))

    df = df.copy()
    df[name] = avgs
    df[name + "_std"] = stds
    df[name + "/element"] = df[name] / df["element_count"]
    return df

def process_results(experiment, results):
    if "algorithm_set" in experiment:
        algorithms = results["algorithm_sets"][experiment["algorithm_set"]]
    else:
        algorithms = experiment["algorithms"]
    
    dataset_results = results["datasets"][experiment["dataset"]]
    info = dataset_results["info"]

    results_per_alg = {}
    for algorithm in algorithms:
        df = pd.DataFrame()

        alg_results = dataset_results["algos"][algorithm]
        xvalues = [row["x"] for row in alg_results]

        df["time_ns"] = [sum(row["times"]) / len(row["times"]) for row in alg_results]
        df["time_ns_std"] = [np.std(row["times"]) for row in alg_results]

        df["selectivity"] = xvalues if info["vary"] == "selectivity" else [info["selectivity"]] * len(xvalues)
        df["selectivity"] = df["selectivity"] / 1000

        df["density"] = xvalues if info["vary"] == "density" else [info["density"]] * len(xvalues)
        df["density"] = df["density"] / 1000

        df["skewness_factor"] = xvalues if info["vary"] == "skew" else [info["skewness_factor"]] * len(xvalues)
        df["skewness_factor"] = df["skewness_factor"] / 1000

        df["max_len_pow"] = xvalues if info["vary"] == "size" else [info["max_len"]] * len(xvalues)
        df["max_len"] = 2 ** df["max_len_pow"]
        df["set_count"] = xvalues if info["vary"] == "set_count" else [info["set_count"]] * len(xvalues)

        def elements_in(set_idx, skewness_factor, max_len):
            return int(max_len / pow(set_idx+1, skewness_factor))
        
        def element_count(row):
            skew = row["skewness_factor"]
            max_len = row["max_len"]
            set_count = int(row["set_count"])
            return sum([elements_in(set_idx, skew, max_len) for set_idx in range(set_count)])

        df["element_count"] = df.apply(element_count, axis=1)
        df["element_bytes"] = df["element_count"] * ELEMENT_BYTES
        df["element_bytes_pow"] = np.log2(df["element_bytes"])
        df["time_s"] = df["time_ns"] / 1e9
        df["throughput_eps"] = df["element_count"] / df["time_s"]

        df["time_ns/element"] =  df["time_ns"] / df["element_count"]
        df["time_s/element"] =  df["time_s"] / df["element_count"]

        for cache in ["l1d", "l1i", "ll"]:
            for stat in ["rd_access", "rd_miss", "wr_access", "wr_miss"]:
                df = add_cpu_stat(df, alg_results, f"{cache}_{stat}", lambda row: row[cache][stat])
            for op in ["rd", "wr"]:
                cache_prefix = f"{cache}_{op}"
                cache_miss = df.get(f"{cache_prefix}_miss")
                cache_access = df.get(f"{cache_prefix}_access")
                if cache_miss is not None and cache_access is not None:
                    df[f"{cache_prefix}_miss_rate"] = cache_miss / cache_access

        for stat in ["branches", "branch_misses", "cpu_stalled_front", "cpu_stalled_back",
                     "instructions", "cpu_cycles", "cpu_cycles_ref"]:
            df = add_cpu_stat(df, alg_results, f"{stat}", lambda row: row[stat])

        df["branch_miss_rate"] = df["branch_misses"] / df["branches"]
        df["ipc"] = df["instructions"] / df["cpu_cycles"]
        df["cpi"] = df["cpu_cycles"] / df["instructions"]

        results_per_alg[algorithm] = df

    return results_per_alg

def relative_throughput(results_per_algo, relative_to, df):
    return df["throughput_eps"] / results_per_algo[relative_to]["throughput_eps"]

def with_relative_throughput(results_per_algo):
    result = {}
    for algo, df in results_per_algo.items():
        result[algo] = df.copy()
        for relative_to in results_per_algo.keys():
            result[algo]["throughput_vs_" + relative_to] = relative_throughput(results_per_algo, relative_to, df)
    return result


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN_PATH
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_PATH

    results_path = Path(results_path)
    out_path = Path(out_path)

    results_file = open(results_path, "r")
    raw_results = json.loads(results_file.read())

    experiments = raw_results["experiments"]
    for experiment in experiments:

        results = process_results(experiment, raw_results)
        results = with_relative_throughput(results)

        exp_dir = out_path / experiment["name"]
        os.makedirs(exp_dir, exist_ok=True)
        for alg,df in results.items():
            alg_file = exp_dir / (alg + ".csv")
            df.to_csv(alg_file, index=False)
            print(alg_file)

if __name__ == "__main__":
    main()
