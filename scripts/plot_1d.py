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

PAIR_VARIABLES = {
    "skew", "density", "selectivity", "max_set_size", "datatype", "distribution"
}
FLOAT_VARIABLES = {"skew", "density", "selectivity"}
SAMPLE_VARIABLES = {
    "skew", "density", "selectivity", "max_set_size", "datatype", 
    "data_distribution", "query_size", "query_distribution", "corpus_size", 
    "corpus_distribution"
}

CCOUNT = 3
CNAMES = ('cycles', 'cache_misses', 'branch_misses')
PROPORTIONS = (
    lambda val, ref: ref / val,
    lambda val, ref: val / ref,
    lambda val, ref: val / ref,
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
    parser.add_argument("-x", "--width", default=10, type=float, help="Image width in inches.")
    parser.add_argument("-y", "--height", default=10, type=float, help="Image height in inches.")

    args = parser.parse_args()

    results_path = pathlib.Path(args.results)
    with open(results_path, "r") as data_file:
        results = json.load(data_file)

    description_path = pathlib.Path(args.description)
    with open(description_path, "r") as description_file:
        description = json.load(description_file)

    note = f"Note: {'' if not 'note' in results else results['note']}"

    # We output the graphs to the same directory as the input results file
    os.chdir(results_path.parents[0])

    experiment = results['experiment']
    algorithms = results['algorithms']
    note = results['note']

    if algorithms[0] in algorithms[1:]:
        control_index = algorithms[1:].index(algorithms[0]) + 1
        algorithms[control_index] = algorithms[control_index] + ' (control)'
    algorithms[0] = algorithms[0] + ' (reference)'

    parameters = description['parameters']
    variables = PAIR_VARIABLES if parameters['type'] == 'pair' else SAMPLE_VARIABLES 

    if args.variable not in variables:
        raise ValueError(f"\"{args.variable}\" is not a valid variable for {paremeters['type']} data.")

    bin_sets = databin_sets(parameters, args.variable, len(description['databins']))
    labels = sorted(list(parameters[args.variable].keys()))
    if args.variable in FLOAT_VARIABLES:
        labels = [f'{float(x):.2g}' for x in labels]

    data = []
    for repeat in results['repeats']:
        per_repeat = []
        for databin in repeat['databins']:
            per_databin = []
            for trial in databin['trials']:
                # order = np.array(trial['order'])
                cycles = np.array(trial['cycles'])
                ll_cache_misses = np.array(trial['ll_cache_misses'])
                branch_misses = np.array(trial['branch_misses'])
                per_databin += [np.stack((cycles, ll_cache_misses, branch_misses))]
            per_repeat += [np.stack(per_databin)]
        data += [np.stack(per_repeat)]
    data = np.stack(data)

    # Data is now collated with dimensions as follows:
    # 0 - repeat
    # 1 - databin
    # 2 - trial
    # 3 - counter
    # 4 - algorithm
    # But we want it as follows:
    # 0 - counter
    # 1 - algorithm
    # 2 - databin
    # 3 - trial
    # 4 - repeat
    data = data.transpose((3, 4, 1, 2, 0))

    for i, (params, bins) in enumerate(bin_sets):
        bin_data = data[:, :, bins]
        plot(
            i,
            experiment,
            algorithms, 
            args.variable, 
            params,
            labels, 
            bin_data, 
            results_path.stem, 
            args.width, 
            args.height, 
            note
        )


def databin_sets(parameters: dict[str, int], variable: str, bin_count: int) -> list[list[int]]:
    # create a list of sets where each set holds all of the databin indices
    # where the given variable is the only thing varying
    i = [({}, set(range(bin_count)))]
    for parameter, index_map in parameters.items():
        if parameter == variable or parameter == 'type':
            continue
        keys = sorted(list(index_map.keys())) 
        values = [({parameter: k}, set(index_map[k])) for k in keys]
        i = [(x[0] | y[0], x[1] & y[1]) for (x, y) in product(i, values)]
    ret = [(v[0], sorted(list(v[1]))) for v in i]
    return ret


def plot(
    subset_index : int, 
    experiment   : str, 
    algorithms   : list[str],
    variable     : str,
    params       : dict[str, str],
    labels       : list[str], 
    data         : np.ndarray,
    fname_base   : str, 
    width        : float, 
    height       : float, 
    note         : str,
):
    prop_algorithms = algorithms[1:]

    # Setup plot figure
    fig = plt.figure(layout='constrained', figsize=(width, height))
    gs = fig.add_gridspec(3, len(CNAMES), height_ratios=[10, 10, 1], hspace=0.1)

    # Rows for proportional data, absolute data, and notes
    prop_axs = [fig.add_subplot(gs[0, i]) for i in range(len(CNAMES))]
    abs_axs = [fig.add_subplot(gs[1, i]) for i in range(len(CNAMES))]
    note_ax = fig.add_subplot(gs[2, :])

    note_ax.set_axis_off()
    note_ax.text(0, 0.5, note + '\n' + str(params)) 

    # Data layout reference
    # 0 - counter
    # 1 - algorithm
    # 2 - databin
    # 3 - trial
    # 4 - repeat

    # Handle calculation and plotting of proportional data
    for counter_index, counter_data in enumerate(data):
        propf = PROPORTIONS[counter_index]
        ax = prop_axs[counter_index]
        cname = CNAMES[counter_index]
        ylabel = YAXIS[counter_index]

        # Prevent division by 0 (only sound for values that are almost never 0)
        counter_data = counter_data.copy()
        counter_data[counter_data == 0] = 1

        # Split reference and to-be-compared data and calculate proportional data
        ref_data = counter_data[0:1]
        cmp_data = counter_data[1:]
        proportions = propf(cmp_data, ref_data)

        # Calculate and plot averages and confidence intervals
        for algo_index, algo_data in enumerate(proportions):
            algo_name = prop_algorithms[algo_index]
            averages = []
            upper = []
            lower = []
            for db_data in algo_data:
                db_data = np.reshape(db_data, -1)
                averages += [np.average(db_data)]
                bootstrap = stats.bootstrap((db_data,), np.average, n_resamples=2000)
                upper += [bootstrap.confidence_interval.high]
                lower += [bootstrap.confidence_interval.low]
            ax.plot(labels, averages, label=algo_name)
            ax.fill_between(labels, lower, upper, alpha=0.5)

        # Configure axis
        ax.set_title(f'Algorithm {cname} relative to {algorithms[0]}')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(f'{variable.capitalize()}')
        low, high = ax.get_ylim()
        ax.set_ylim(0, max(1, high))
        ax.legend()

    # Handle calculation and plotting of absolute data
    for counter_index, counter_data in enumerate(data):
        ax = abs_axs[counter_index]
        cname = CNAMES[counter_index]

        # Calculate and plot averages and confidence intervals
        for algo_index, algo_data in enumerate(counter_data):
            algo_name = algorithms[algo_index]
            averages = []
            upper = []
            lower = []
            for db_data in algo_data:
                db_data = np.reshape(db_data, -1)
                averages += [np.average(db_data)]
                bootstrap = stats.bootstrap((db_data,), np.average, n_resamples=2000)
                upper += [bootstrap.confidence_interval.high]
                lower += [bootstrap.confidence_interval.low]
            ax.plot(labels, averages, label=algo_name)
            ax.fill_between(labels, lower, upper, alpha=0.5)

        # Configure axis
        ax.set_title(f'Algorithm {cname}')
        ax.set_ylabel('Counts')
        ax.set_xlabel(f'{variable.capitalize()}')
        low, high = ax.get_ylim()
        ax.set_ylim(0, max(1, high))
        ax.legend()

    filename = f'{fname_base}.{experiment}.{variable}.{subset_index}.counters.png'
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main()
