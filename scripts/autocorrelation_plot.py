'''
This script provides functionality for aggregating results per run and plotting
this aggregate to allow for visual inspection of the effect of cache and
other micro-architectural warmup on algorithm performance.

This aggregation can occur over the entire set of measurements or the
measurements can be binned by absolute runtime to allow for inspection at
different timescales. This can be achieved with the --bins option.

To aid in this binning this script can also produce a graph of the runtimes
in sorted order with the --runtimes option. To view this graph with a log y-axis
use the --log argument.

If the outliers on the boxplots are too large to do proper comparisons then the
--bound option can be adjusted to change the limits on the y-axis so that the
IQR boxes and whiskers can actually be seen in the plot.
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
        description = "Provides visualisation of within-trial run distribution."
    )
    parser.add_argument("results", help="Path to json results file.")
    parser.add_argument("-k", "--lag", default=1, type=int, help="Autocorrelation lag.")
    parser.add_argument("-x", "--width", default=10, type=int, help="Image width in inches.")
    parser.add_argument("-y", "--height", default=10, type=int, help="Image height in inches.")
    args = parser.parse_args()

    results_path = pathlib.Path(args.results)
    with open(results_path, "r") as data_file:
        results = json.load(data_file)

    # We output the graphs to the same directory as the input results file
    os.chdir(results_path.parents[0]) 

    tsc_overhead = results["tsc_characteristics"]["overhead"]

    # We collate every measurement in every trial and normalize them relative to the
    # trial median and IQR
    per_experiment = {}
    for experiment_result in results["experiment_results"]:
        experiment_name = experiment_result["experiment_name"]
        per_algorithm = {}
        for algorithm_result in experiment_result["algorithm_results"]:
            algorithm_name = algorithm_result["algorithm_name"]
            measurements = []
            for repeat_result in algorithm_result["repeat_results"]:
                for databin_result in repeat_result["databin_results"]: 
                    if "pair" in databin_result["results"]:
                        for trial_result in databin_result["results"]["pair"]:
                            deltas = np.array(trial_result["deltas"]) - tsc_overhead
                            acs = autocorrelation(deltas, args.lag)
                            measurements.append(acs)
                    else:
                        raise NotImplementedError("sample")
            per_algorithm[algorithm_name] = measurements
        per_experiment[experiment_name] = per_algorithm

    for e_name, e_data in per_experiment.items():
        for a_name, a_data in e_data.items():
            measurements = [np.median(x) for x in zip(*a_data)]
            e_data[a_name] = measurements

    # Per-experiment plotting
    plt.rcParams['figure.constrained_layout.use'] = True
    for e_name, e_data in per_experiment.items():
        fig, axs = plt.subplots(len(e_data), 1, layout='constrained', squeeze=False, sharey=True, sharex=True)
        fig.supxlabel("Lag")
        fig.supylabel("Trial Autocorrelation")
        for ax, (a_name, a_data) in zip(axs.flatten(), e_data.items()):
            ax.bar(np.arange(1, len(a_data)+1), a_data)
            ax.set_title(a_name)
            ax.grid(visible=True, which="both", axis="y")
        data = list(e_data.values())
        labels = list(e_data.keys())

        fig.set_size_inches(args.width, args.height)
        fig.suptitle(f"Experiment: {e_name}")

        path = f'{results_path.stem}.{e_name}.autocorrelation.png'
        plt.savefig(path)
        plt.close()


def autocorrelation(data, k_max):
    average = np.average(data)
    denom = np.sum((data - average)**2)
    if denom == 0:
        return [0] * k_max
    ks = []
    dmk = data - average
    for k in range(1, k_max+1):
        numer = np.sum(dmk[:-k] * dmk[k:])
        ks += [numer / denom]
    return ks


if __name__ == "__main__":
    main()
