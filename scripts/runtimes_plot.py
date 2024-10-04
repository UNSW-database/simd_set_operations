'''
TODO
'''
import argparse
import pathlib
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


def main():
    parser = argparse.ArgumentParser(
        description = "Generate graphs of sorted experiment runtimes."
    )
    parser.add_argument("results", help="Path to json results file.")
    parser.add_argument("-b", "--bins", type=int, help="Number of bins in histogram.")
    parser.add_argument("-x", "--width", default=10, type=int, help="Image width in inches.")
    parser.add_argument("-y", "--height", default=10, type=int, help="Image height in inches.")
    args = parser.parse_args()

    bins = 'auto' if args.bins is None else args.bins

    results_path = pathlib.Path(args.results)
    with open(results_path, "r") as data_file:
        results = json.load(data_file)

    # We output the graphs to the same directory as the input results file
    os.chdir(results_path.parents[0]) 

    tsc_overhead = results["tsc_characteristics"]["overhead"]

    # Collate all of the measurements grouped by experiment and algorithm
    measurements = {}
    for experiment_result in results["experiment_results"]:
        experiment_name = experiment_result["experiment_name"]
        experiment_measurements = {}
        for algorithm_result in experiment_result["algorithm_results"]:
            algorithm_name = algorithm_result["algorithm_name"]
            algorithm_measurements = []
            for repeat_result in algorithm_result["repeat_results"]:
                for databin_result in repeat_result["databin_results"]: 
                    if "pair" in databin_result["results"]:
                        for trial_result in databin_result["results"]["pair"]:
                            deltas = np.array(trial_result["deltas"]) - tsc_overhead
                            algorithm_measurements.extend(deltas)
                    else:
                        raise NotImplementedError("sample")
            experiment_measurements[algorithm_name] = algorithm_measurements
        measurements[experiment_name] = experiment_measurements

    # Create 2 by 1 graphs of combined and per-algorithm histograms
    for e_name, algo_data_map in measurements.items():
        data = list(algo_data_map.values())
        labels = list(algo_data_map.keys())
        edges = 10 ** np.histogram_bin_edges(np.log10(data), bins=bins)

        fig, ax = plt.subplots()
        fig.set_size_inches(args.width, args.height)
        fig.suptitle(f"Experiment: {e_name}")

        # We convert the runtimes to a histogram as we want to visualise the distribution
        ax.hist(data, bins=edges, histtype='barstacked', label=labels)
        ax.legend(loc='best')
        ax.set_title("Runtime Histogram")
        ax.set_xlabel("Runtime (TSC Cycles)")
        ax.set_ylabel("Bin Count")
        ax.set_xscale('log')
        ax.grid(visible=True, which="both", axis="x")
        ax.xaxis.set_minor_formatter(plticker.ScalarFormatter())

        plt.savefig(f"{results_path.stem}.{e_name}.runtimes.png")
        plt.close()


if __name__ == "__main__":
    main()