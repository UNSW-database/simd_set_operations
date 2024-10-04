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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

def main():
    parser = argparse.ArgumentParser(
        description = "Provides visualisation of within-trial run distribution."
    )
    parser.add_argument("results", help="Path to json results file.")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins in histogram.")
    parser.add_argument("--bound", type=float, help="Symmetric bound of trial-median relative runtime used to set axis limits.")
    parser.add_argument("--cutoff", type=float, help="Outlier cutoff.")
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
                            med= median(deltas)
                            # iqr = np.subtract(*np.percentile(deltas, (75, 25)))
                            rrs = (deltas / med) - 1
                            if args.cutoff:
                                for rr in rrs:
                                    if abs(rr) <= args.cutoff:
                                        measurements.append(rr)
                            else:
                                measurements.extend(rrs)
                    else:
                        raise NotImplementedError("sample")
            per_algorithm[algorithm_name] = measurements
        per_experiment[experiment_name] = per_algorithm

    # Per-experiment plotting
    plt.rcParams['figure.constrained_layout.use'] = True
    for e_name, e_data in per_experiment.items():
        fig, axs = plt.subplots(len(e_data), 1, layout='constrained', squeeze=False, sharey=True, sharex=True)
        fig.supxlabel("Bin count")
        fig.supylabel(" Median-relative runtime")
        for ax, (a_name, a_data) in zip(axs.flatten(), e_data.items()):
            data_range = None if args.bound is None else (1 - args.bound, 1 + args.bound)
            ax.hist(a_data, bins=args.bins, range=data_range, histtype='bar', orientation='horizontal')
            ax.set_title(a_name)
            ax.grid(visible=True, which="both", axis="x")
            if args.bound:
                ax.set_ylim((1-args.bound, 1+args.bound))
        data = list(e_data.values())
        labels = list(e_data.keys())

        fig.set_size_inches(args.width, args.height)
        fig.suptitle(f"Experiment: {e_name}")

        path = f'{results_path.stem}.{e_name}.outlier_run.png'
        plt.savefig(path)
        plt.close()


def median(a):
    return np.sort(a)[len(a) // 2]


if __name__ == "__main__":
    main()
