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
import matplotlib
import colorsys

def main():
    parser = argparse.ArgumentParser(
        description = "Provides visualisation and removal of initial cold-cache datapoints."
    )
    parser.add_argument("results", help="Path to json results file.")
    parser.add_argument("--bound", help="Symmetric bound of trial-median relative runtime used to set axes limits.")
    parser.add_argument("--bins", nargs="*", default=[], help="Runtime breakpoints for splitting data into bins.")
    parser.add_argument("--algorithms", action="store_true", help="Make per-algorithm rather than fully collated graphs.")
    parser.add_argument("--cutoff", default=100, type=int, help="Set the cutoff point for total datapoints under which a graph will be colored red.")
    parser.add_argument("-x", "--width", default=10, type=int, help="Image width in inches.")
    parser.add_argument("-y", "--height", default=10, type=int, help="Image height in inches.")
    args = parser.parse_args()

    bound = None if args.bound is None else float(args.bound)
    bins = sorted([float(x) for x in args.bins])
    extended_bins = [-math.inf] + bins + [math.inf]
    bin_windows = list(zip((x for x in extended_bins[:-1]), (y for y in extended_bins[1:])))

    results_path = pathlib.Path(args.results)
    with open(results_path, "r") as data_file:
        results = json.load(data_file)

    # We output the graphs to the same directory as the input results file
    os.chdir(results_path.parents[0]) 

    tsc_overhead = results["tsc_characteristics"]["overhead"]

    # We collate and normalize relative to the trial median every measurement in every trial
    measurement_bins = {}
    for experiment_result in results["experiment_results"]:
        experiment_name = experiment_result["experiment_name"]
        experiment_bins = {}
        for algorithm_result in experiment_result["algorithm_results"]:
            algorithm_name = algorithm_result["algorithm_name"]
            algorithm_bins = [{} for _ in range(len(bins) + 1)] 
            for repeat_result in algorithm_result["repeat_results"]:
                for databin_result in repeat_result["databin_results"]: 
                    if "pair" in databin_result["results"]:
                        for trial_result in databin_result["results"]["pair"]:
                            deltas = np.array(trial_result["deltas"]) - tsc_overhead
                            median = np.median(deltas)
                            rrs = deltas / median
                            bin_index = np.digitize([median], bins)[0]
                            for i, rr in enumerate(rrs):
                                algorithm_bins[bin_index].setdefault(i, list()).append(rr)
                    else:
                        raise NotImplementedError("sample")
            experiment_bins[algorithm_name] = algorithm_bins
        measurement_bins[experiment_name] = experiment_bins

    # if --algorithms is not specified then we collate the per-algorithm data
    if not args.algorithms:
        for e_name, e_data in measurement_bins.items():
            e_all = [{} for _ in range(len(bins) + 1)]
            for a_data in e_data.values():
                for a_bin, all_bin in zip(a_data, e_all):
                    for i, rrs in a_bin.items():
                        all_bin.setdefault(i, list()).extend(rrs)
            measurement_bins[e_name] = {'combined': e_all}


    # Per-experiment plotting
    plt.rcParams['figure.constrained_layout.use'] = True
    for e_name, e_data in measurement_bins.items():
        # find largest per algorithm binsize to calculate a color norm
        largest_binsize = 0
        for a_data in e_data.values():
            for bin_ in a_data:
                if 0 in bin_ and len(bin_[0]) > largest_binsize:
                    largest_binsize = len(bin_[0])
        norm = matplotlib.colors.TwoSlopeNorm(args.cutoff, vmin=0, vmax=largest_binsize)

        # We box plot the measurements with measurement # on the x-axis and 
        # trial median relative runtime on the y-axis

        algorithm_count = len(e_data)
        bin_count = len(bins) + 1
        fig, axs = plt.subplots(algorithm_count, bin_count, sharex=True, sharey=True, squeeze=False, layout='constrained')

        fig.set_size_inches(args.width, args.height)
        fig.suptitle(f"Run No. vs. Runtime | Experiment: {e_name}", y=0.999)
        fig.supylabel("Median Relative Runtime", x=0.01)
        fig.supxlabel("Run No.")

        for i, (a_name, a_data) in enumerate(e_data.items()):
            for bin_index, (lo, hi) in enumerate(bin_windows):
                data = a_data[bin_index]
                data = [data[x] for x in range(len(data))]
                labels = [str(x) for x in range(len(data))]

                ax = axs[i,bin_index]

                # Draw the graph, but only if we have data
                if data != []:
                    size = len(data[0])
                    ax.boxplot(data, labels=labels)
                    ax.grid(visible=True, which="both", axis="y")
                    if bound:
                        ax.set_ylim([1 - bound, 1 + bound])
                else:
                    size = 0
                    
                # Color the graphs based on datapoint count
                color = matplotlib.colormaps['RdYlGn'](norm(size))
                h, l, s = colorsys.rgb_to_hls(*color[:-1])
                lightened = colorsys.hls_to_rgb(h, 0.85, s)
                ax.set_facecolor(lightened)

                # Set row and column titles
                if i == 0:
                    ax.set_title(f"Bin #{bin_index} | {lo:.0e} to {hi:.0e} cycles", fontsize='small')
                if bin_index == 0:
                    ax.set_ylabel(f"{a_name}", fontsize='small')

        path = f'{results_path.stem}.{e_name}.warmup.png'
        plt.savefig(path)
        plt.close()


if __name__ == "__main__":
    main()