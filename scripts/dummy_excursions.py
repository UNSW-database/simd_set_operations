import argparse
import pathlib
import json
import os
import math
import numpy as np


def main():
    parser = argparse.ArgumentParser(description = "Find largest deviation from expected runtime in dummy runs.")
    parser.add_argument("results", help="Path to json results file.")
    args = parser.parse_args()

    results_path = pathlib.Path(args.results)
    with open(results_path, "r") as data_file:
        results = json.load(data_file)

    # Find the dummy experiment
    for experiment in results["experiment_results"]:
        if experiment["experiment_name"] == "dummy":
            dummy_experiment = experiment
            break

    # Go through all measurements and collate per dummy algo
    data_types = ["cycles", "cache_misses", "branch_misses", "page_faults", 
        "context_switches", "cpu_migrations"]
    data = {}
    for algorithm in dummy_experiment["algorithm_results"]:
        algorithm_name = algorithm["algorithm_name"]
        data[algorithm_name] = {t: [] for t in data_types}
        for repeat in algorithm["repeat_results"]:
            for databin in repeat["databin_results"]:
                for trial in databin["results"]["pair"]:
                    for t in data_types:
                        data[algorithm_name][t].extend(trial[t])

    for t in data_types:
        print(f"Statistics ({t}):")
        for algo_name, d in data.items():
            values = np.array(d[t])
            average = np.average(values)
            std_dev = np.std(values)
            max_val = np.max(values)
            min_val = np.min(values)
            median = np.median(values)
            out_pct = np.sum(values > 1.1 * average) / len(values) * 100
            print(f"    {algo_name}: average[{average:.1f}], std_dev[{std_dev:.1f}], median[{median}], 10% outliers[{out_pct:0.1f}%], min[{min_val}], max[{max_val}]")


if __name__ == "__main__":
    main()
