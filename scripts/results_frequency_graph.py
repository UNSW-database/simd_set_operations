import argparse
import pathlib
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

NS = 1_000_000_000
US = 1_000_000
MS = 1_000

def main():
    parser = argparse.ArgumentParser(
        description = "Graphs the frequency data collected during result \
            collection from the SIMD set operations benchmark suite.",
    )
    parser.add_argument("results_file", help="Path to json results file.")
    parser.add_argument("-s", "--start", help="Start time, in seconds.", type=float)
    parser.add_argument("-e", "--end", help="End time, in seconds.", type=float)
    args = parser.parse_args()

    if (args.start == None or args.end == None) and args.start != args.end:
        raise ValueError("If start or end are specified then both must be specified.")

    results_path = pathlib.Path(args.results_file)
    with open(results_path, "r") as data_file:
        results = json.load(data_file)

    # We output the graphs to the same directory as the input results file
    os.chdir(results_path.parents[0]) 
    
    (time_deltas_us, frequencies_hz) = results_to_frequencies(results)

    time_deltas_secs = time_deltas_us / US
    frequencies_ghz = frequencies_hz / NS
    if args.start != None:
        filter = np.logical_and(args.start <= time_deltas_secs, time_deltas_secs <= args.end)
        time_deltas_secs = time_deltas_secs[filter]
        frequencies_ghz = frequencies_ghz[filter]

    error = calc_error(frequencies_ghz, results)
    plot_max = np.max(frequencies_ghz) + 0.2

    filename = results_path.stem + ".frequencies.png"

    plot_frequencies(time_deltas_secs, frequencies_ghz, error, plot_max, filename)


def results_to_frequencies(results):
    measurements = []
    for experiment in results["experiment_results"]:
        for algorithm in experiment["algorithm_results"]:
            for repeat in algorithm["repeat_results"]:
                for databin in repeat["databin_results"]:
                    results_type = databin["results"]
                    if "pair" in results_type:
                        samples = [results_type["pair"]]
                    else:
                        samples = results_type["sample"]
                    for sample in samples:
                        for trial in sample:
                            measurements.append(trial["pre"])
                            measurements.append(trial["post"])

    cycles_counts = np.array([x["cc"] for x in measurements])
    time_deltas = np.array([x["td"] for x in measurements])
    
    tscc = results["tsc_characteristics"]
    overhead = tscc["overhead"]
    ref_cycles = results["reference_cycles"]
    ref_freq = tscc["frequency"]
    frequencies = (ref_freq * ref_cycles) / (cycles_counts - overhead)

    return (time_deltas, frequencies)


def calc_error(frequencies, results):
    raw_error = np.array(results["tsc_characteristics"]["error"])
    prop_error = raw_error / results["reference_cycles"]
    lower = frequencies * (1 - prop_error[0])
    upper = frequencies * (1 + prop_error[1])
    return (lower, upper)


def plot_frequencies(time_deltas, frequencies, error, plot_max, filename):
    fig, ax = plt.subplots()

    fig.set_size_inches(30, 10)

    ax.plot(time_deltas, frequencies)
    plt.fill_between(time_deltas, error[0], error[1], alpha=0.5)

    ax.set_ylim(ymin=0, ymax=plot_max)
    ax.set_title(f"Results CPU Frequency")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_xlabel("Time (s)")
    loc = plticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    ax.grid(visible=True, which="major", axis="y")

    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main()