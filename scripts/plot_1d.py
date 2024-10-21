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

def main():
    parser = argparse.ArgumentParser(
        description = "Graphs relative performance of algorithms for a single variable."
    )
    parser.add_argument("results", help="Path to json results file.")
    parser.add_argument("description", help="Path to databin description.")
    parser.add_argument("variable", help="Variable to graph.")
    parser.add_argument("reference", help="Reference algorithm.")
    parser.add_argument("-x", "--width", default=10, type=int, help="Image width in inches.")
    parser.add_argument("-y", "--height", default=10, type=int, help="Image height in inches.")

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

    # We output the graphs to the same directory as the input results file
    os.chdir(results_path.parents[0])

    # TODO
    # NB: Data cleaning should be done separately
    # For the selected variable, create bins of databins that only vary in that variable
    # Graph each bin

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
                collated_bins = {i: {'cycles': []} for i in subset}
                for repeat in algorithm['repeat_results']:
                    databins = repeat['databin_results']
                    for i, d in collated_bins.items():
                        # calculate per-trial average
                        for trial in databins[i]['results']['pair']:
                            for counter in d:
                                d[counter].extend(trial[counter])
                per_algorithm[algorithm['algorithm_name']] = collated_bins
            per_subset.append(per_algorithm)
        per_experiment[experiment['experiment_name']] = per_subset


    for e_name, e_data in per_experiment.items():
        for si, s_data in enumerate(e_data):
            plot(e_name, si, s_data, args.variable, args.reference, labels[si], results_path.stem, args.width, args.height)


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


def plot(experiment_name, subset_index, subset_data, variable_name, reference_algorithm, labels, fname_base, width, height):
    labels = list(labels.values())
    # subset_data = {algorithm_name: collated_bins}
    #   collated_bins = {i: {'cycles': []} for i in subset}

    # Split reference and to-be-compared data
    ref_data = subset_data[reference_algorithm]
    ref_avgs = {i: {c: np.average(d) for c, d in counters.items()} for i, counters in ref_data.items()}
    cmp_data = {a: d for a, d in subset_data.items() if a != reference_algorithm}

    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)

    for a_name in cmp_data:
        proportions = {}
        bootstraps = {}
        collated = cmp_data[a_name]
        for i in collated:
            proportions[i] = {}
            bootstraps[i] = {}

            counters = collated[i]
            ref_counters = ref_data[i]
            ref_counter_averages = ref_avgs[i]

            for cname in counters:
                values = counters[cname]
                average = np.average(values)
                ref_values = ref_counters[cname]
                ref_average = ref_counter_averages[cname]

                proportions[i][cname] = ref_average / average
                def proportion(values, ref_values):
                    return np.average(ref_values) / np.average(values)
                bootstraps[i][cname] = stats.bootstrap((values, ref_values), proportion, n_resamples=2000)

        proportion, upper, lower = [], [], []
        for i in proportions:
            proportion.append(proportions[i]['cycles'])
            lower.append(bootstraps[i]['cycles'].confidence_interval.low)
            upper.append(bootstraps[i]['cycles'].confidence_interval.high)

        ax.plot(labels, proportion, label=a_name)
        ax.fill_between(labels, lower, upper, alpha=0.5)

    # Plot data
    ax.set_title(f'Algorithm speedup relative to {reference_algorithm}')
    ax.set_ylabel('Speedup (ref. cycles / algo. cycles)')
    ax.set_xlabel(f'{variable_name.capitalize()}')
    low, high = ax.get_ylim()
    ax.set_ylim(0, max(1, high))
    ax.legend()

    filename = f'{fname_base}.{experiment_name}.{subset_index}.{variable_name}.cycles.png'
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main()
