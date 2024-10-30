import argparse
import pathlib
import json
import os
from itertools import product
from copy import deepcopy
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

NS = 1_000_000_000
US = 1_000_000
MS = 1_000

VARIABLES = {"skew", "density", "selectivity", "size", "datatype", "distribution"}

CNAMES = ('cycles', 'cache_misses', 'branch_misses')
PROPORTIONS = (
    lambda val, ref: np.average(ref) / np.average(val),
    lambda val, ref: np.average(val) / np.average(ref),
    lambda val, ref: np.average(val) / np.average(ref),
)
YAXIS = (
    'Speedup (ref. value / value)', 
    'Ratio (value / ref. value)',
    'Ratio (value / ref. value)',
)


def main():
    parser = argparse.ArgumentParser(
        description = "Graphs relative performance of algorithms for a single variable."
    )
    parser.add_argument("results", help="Path to json results file.")
    parser.add_argument("description", help="Path to databin description.")
    parser.add_argument("variable", help="Variable to graph.")
    parser.add_argument("reference", help="Reference algorithm.")
    parser.add_argument("-rw", default=0, type=int, help="Remove this many values from the start of each trial.")
    parser.add_argument("-x", "--width", default=10, type=float, help="Image width in inches.")
    parser.add_argument("-y", "--height", default=10, type=float, help="Image height in inches.")

    args = parser.parse_args()

    if args.variable not in VARIABLES:
        raise ValueError(f"\"{args.variable}\" is not a valid variable.")

    results_path = pathlib.Path(args.results)
    with open(results_path, "r") as data_file:
        results = json.load(data_file)

    if args.reference not in algorithms(results):
        raise ValueError(f"\"{args.reference}\" is not a valid algorithm.")

    description_path = pathlib.Path(args.description)
    with open(description_path, "r") as description_file:
        description = json.load(description_file)

    note = f"Note: {'' if not 'note' in results else results['note']}"

    # We output the graphs to the same directory as the input results file
    os.chdir(results_path.parents[0])

    bins = databin_bins(description, args.variable)
    labels = bins_labels(bins, description, args.variable)

    # Collate data
    # per_experiment = {e_name: per_subset}
    #   per_subset = [per_algorithm]
    #     per_algorithm = {a_name: collated_bins}
    #       collated_bins = {i: {'cycles': []} for i in subset}
    per_experiment = {}
    for experiment in results['experiment_results']:
        per_subset = []
        for si, subset in enumerate(bins):
            per_algorithm = {}
            for algorithm in experiment["algorithm_results"]:
                # just dump everything into one
                collated_bins = {i: {cn: [] for cn in CNAMES} for i in subset}
                for repeat in algorithm['repeat_results']:
                    databins = repeat['databin_results']
                    for databin in databins:
                        if databin is None:
                            continue
                        i = int(databin['databin_index'])
                        collated_bin = collated_bins[i]
                        for trial in databin['results']['pair']:
                            for counter in CNAMES:
                                collated_bin[counter].extend(trial[counter][args.rw:])
                per_algorithm[algorithm['algorithm_name']] = collated_bins
            per_subset.append(per_algorithm)
        per_experiment[experiment['experiment_name']] = per_subset

    for e_name, e_data in per_experiment.items():
        for si, s_data in enumerate(e_data):
            plot(e_name, si, s_data, args.variable, args.reference, labels[si], results_path.stem, args.width, args.height, note)


def bins_labels(bins, description, variable):
    # Convert bin numbers to variable labels
    labels = []
    for subset in bins:
        subset_labels = {}
        for bin_index in subset:
            desc = description[bin_index]
            match variable:
                case "selectivity":
                    lengths = desc['lengths']
                    selectivity = lengths['intersection_length'] / min(lengths['set_lengths'])
                    subset_labels[bin_index] = selectivity
        labels.append(subset_labels)
    return labels


def algorithms(results: dict) -> set[str]:
    names_lists = [[algorithm["algorithm_name"] for algorithm in experiment["algorithm_results"]] for experiment in results["experiment_results"]]
    return {name for names in names_lists for name in names}


# Find the disjoint sets where each set has only the given variable varying
def databin_bins(description: dict, variable: str) -> list[list[int]]:
    # dictionaries
    datatype = {}
    max_value = {}
    max_length = {}
    min_length = {}
    intersection_length = {}
    distribution = {}
    trials = {}

    # create mapping from values to databin indices
    for i, databin in enumerate(description):
        datatype.setdefault(databin["datatype"], set()).add(i)
        max_value.setdefault(databin["max_value"], set()).add(i)
        if "set_lengths" in databin["lengths"]:
            max_length.setdefault(databin["lengths"]["set_lengths"][0], set()).add(i)
            min_length.setdefault(databin["lengths"]["set_lengths"][1], set()).add(i)
        else:
            raise NotImplementedError("samples")
        intersection_length.setdefault(databin["lengths"]["intersection_length"], set()).add(i)
        distribution.setdefault(databin["distribution"]["type"], set()).add(i)
        trials.setdefault(databin["trials"], set()).add(i)

    # lists of sets of indices for each specific value
    datatype = list(datatype.values())
    max_value = list(max_value.values())
    max_length = list(max_length.values())
    min_length = list(min_length.values())
    intersection_length = list(intersection_length.values())
    distribution = list(distribution.values())
    trials = list(trials.values())

    # create a list of sets where each set holds all of the databin indices
    # where the given variable is the only thing varying
    match variable:
        case "selectivity":
            i = datatype
            i = [x & y for (x, y) in product(i, max_value)]
            i = [x & y for (x, y) in product(i, max_length)]
            i = [x & y for (x, y) in product(i, min_length)]
            i = [x & y for (x, y) in product(i, distribution)]
            i = [x & y for (x, y) in product(i, trials)]
        case _:
            raise NotImplementedError(f"{variable}")

    ret = [sorted(list(v)) for v in i]
    return ret


def plot(experiment_name, subset_index, subset_data, variable_name, reference_algorithm, labels, fname_base, width, height, note):
    labels = list(labels.values())
    # subset_data = {algorithm_name: collated_bins}
    #   collated_bins = {i: {'cycles': []} for i in subset}

    # Split reference and to-be-compared data
    ref_data = subset_data[reference_algorithm]
    cmp_data = {a: d for a, d in subset_data.items() if a != reference_algorithm}

    fig = plt.figure(layout='constrained', figsize=(width, height))
    gs = fig.add_gridspec(3, len(CNAMES), height_ratios=[10, 10, 1], hspace=0.1)

    prop_axs = [fig.add_subplot(gs[0, i]) for i in range(len(CNAMES))]
    abs_axs = [fig.add_subplot(gs[1, i]) for i in range(len(CNAMES))]
    note_ax = fig.add_subplot(gs[2, :])

    note_ax.set_axis_off()
    note_ax.text(0, 0.5, note)

    for a_name in cmp_data:
        proportions = {}
        bootstraps = {}
        collated = cmp_data[a_name]
        for i in collated:
            proportions[i] = {}
            bootstraps[i] = {}

            counters = collated[i]
            ref_counters = ref_data[i]

            for cname, propf in zip(CNAMES, PROPORTIONS):
                values = counters[cname]
                ref_values = ref_counters[cname]

                if len(values) == 0:
                    continue

                proportions[i][cname] = propf(values, ref_values)
                bootstraps[i][cname] = stats.bootstrap((values, ref_values), propf, n_resamples=2000)

        for j, cname in enumerate(CNAMES):
            proportion, upper, lower = [], [], []
            for i in proportions:
                if cname not in proportions[i]:
                    continue
                proportion.append(proportions[i][cname])
                lower.append(bootstraps[i][cname].confidence_interval.low)
                upper.append(bootstraps[i][cname].confidence_interval.high)

            if len(proportion) != len(labels):
                continue
            prop_axs[j].plot(labels, proportion, label=a_name)
            prop_axs[j].fill_between(labels, lower, upper, alpha=0.5)

    # Configure axes
    for i, (cname, ylabel) in enumerate(zip(CNAMES, YAXIS)):
        prop_axs[i].set_title(f'Algorithm {cname} relative to {reference_algorithm}')
        prop_axs[i].set_ylabel(ylabel)
        prop_axs[i].set_xlabel(f'{variable_name.capitalize()}')
        low, high = prop_axs[i].get_ylim()
        prop_axs[i].set_ylim(0, max(1, high))
        prop_axs[i].legend()

    for a_name in subset_data:
        averages = {}
        bootstraps = {}

        collated = subset_data[a_name]
        for i in collated:
            averages[i] = {}
            bootstraps[i] = {}

            counters = collated[i]
            for cname in CNAMES:
                values = counters[cname]
                if len(values) == 0:
                    continue
                averages[i][cname] = np.average(values)
                bootstraps[i][cname] = stats.bootstrap((values, ), np.average, n_resamples=2000)

        for j, cname in enumerate(CNAMES):
            average, upper, lower = [], [], []
            for i in averages:
                if cname not in averages[i]:
                    continue
                average.append(averages[i][cname])
                lower.append(bootstraps[i][cname].confidence_interval.low)
                upper.append(bootstraps[i][cname].confidence_interval.high)

            if len(proportion) != len(labels):
                continue
            abs_axs[j].plot(labels, average, label=a_name)
            abs_axs[j].fill_between(labels, lower, upper, alpha=0.5)

    # Plot data
    for i, cname in enumerate(CNAMES):
        abs_axs[i].set_title(f'Algorithm {cname}')
        abs_axs[i].set_ylabel('Counts')
        abs_axs[i].set_xlabel(f'{variable_name.capitalize()}')
        low, high = abs_axs[i].get_ylim()
        abs_axs[i].set_ylim(0, max(1, high))
        abs_axs[i].legend()

    filename = f'{fname_base}.{experiment_name}.{subset_index}.{variable_name}.counters.png'
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main()
