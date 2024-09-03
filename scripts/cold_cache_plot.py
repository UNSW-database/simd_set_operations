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

NS = 1_000_000_000
US = 1_000_000
MS = 1_000

def main():
    parser = argparse.ArgumentParser(
        description = "Provides visualisation and removal of initial cold-cache datapoints."
    )
    parser.add_argument("results", help="Path to json results file.")
    parser.add_argument("--runtimes", action="store_true", help="Produce a graph of sorted runtimes instead.")
    parser.add_argument("--log", action="store_true", help="Display the runtimes graph with a log y scale.")
    parser.add_argument("--bound", help="Symmetric bound of trial-median relative runtime used to set axes limits.")
    parser.add_argument("--bins", nargs="*", default=[], help="Runtime breakpoints for splitting data into bins.")
    args = parser.parse_args()

    bound = None if args.bound is None else float(args.bound)
    bins = sorted([float(x) for x in args.bins])
    extended_bins = [-math.inf] + bins + [math.inf]

    results_path = pathlib.Path(args.results)
    with open(results_path, "r") as data_file:
        results = json.load(data_file)

    # We output the graphs to the same directory as the input results file
    os.chdir(results_path.parents[0]) 

    tsc_overhead = results["tsc_characteristics"]["overhead"]

    if not args.runtimes:
        # We collate and normalize relative to the trial median every measurement in every trial
        measurement_bins = [{} for _ in range(len(bins) + 1)]
        for experiment_result in results["experiment_results"]:
            for algorithm_result in experiment_result["algorithm_results"]:
                for repeat_result in algorithm_result["repeat_results"]:
                    for databin_result in repeat_result["databin_results"]: 
                        if "pair" in databin_result["results"]:
                            for trial_result in databin_result["results"]["pair"]:
                                deltas = np.array(trial_result["deltas"]) - tsc_overhead
                                median = np.median(deltas)
                                rrs = deltas / median
                                bin_index = np.digitize([median], bins)[0]
                                for i, rr in enumerate(rrs):
                                    measurement_bins[bin_index].setdefault(i, list()).append(rr)
                        else:
                            raise NotImplementedError("sample")

        for bin_index, (lo, hi) in enumerate(zip((x for x in extended_bins[:-1]), (y for y in extended_bins[1:]))):
            measurements = measurement_bins[bin_index]

            # We box plot the measurements with measurement # on the x-axis and 
            # trial median relative runtime on the y-axis
            labels = [str(x) for x in range(len(measurements))]
            data = [measurements[x] for x in range(len(measurements))]

            fig, ax = plt.subplots()
            fig.set_size_inches(10, 10)
            fig.suptitle(f"Trial measurements over time")

            ax.boxplot(data, labels=labels)

            # ax.set_ylim(ymin=0, ymax=plot_max)
            ax.set_title(f"Bin #{bin_index} | {lo} to {hi} TSC Cycles")
            ax.set_xlabel("Measurement #")
            ax.set_ylabel("Runtime relative to trial median")
            if bound:
                ax.set_ylim([1 - bound, 1 + bound])
            # loc = plticker.MultipleLocator(base=bound / 5)
            # ax.yaxis.set_major_locator(loc)
            ax.grid(visible=True, which="both", axis="y")

            plt.savefig(results_path.stem + f".cold_cache.{bin_index}.png")
            plt.close()
    else:
        # We collect every measured runtime
        measurements = []
        for experiment_result in results["experiment_results"]:
            for algorithm_result in experiment_result["algorithm_results"]:
                for repeat_result in algorithm_result["repeat_results"]:
                    for databin_result in repeat_result["databin_results"]: 
                        if "pair" in databin_result["results"]:
                            for trial_result in databin_result["results"]["pair"]:
                                deltas = np.array(trial_result["deltas"]) - tsc_overhead
                                measurements.extend(deltas)
                        else:
                            raise NotImplementedError("sample")

        # We sort the runtimes as we want to visualise the shape of the runtimes
        xs = np.arange(len(measurements))
        ys = np.array(sorted(measurements))

        # We plot the 
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)

        if args.log:
            ax.semilogy(xs, ys)
        else:
            ax.plot(xs, ys)

        # ax.set_ylim(ymin=0, ymax=plot_max)
        ax.set_title(f"Sorted measurements")
        ax.set_xlabel("Measurement #")
        ax.set_ylabel("Runtime (TSC Cycles)")
        ax.grid(visible=True, which="both", axis="y")
        ax.yaxis.set_minor_formatter(plticker.ScalarFormatter())

        plt.savefig(results_path.stem + ".runtimes.png")
        plt.close()




if __name__ == "__main__":
    main()