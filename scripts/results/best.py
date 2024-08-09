# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import label

RESULTS = Path("../../processed/")
mpl.rcParams['figure.dpi'] = 300

MEMS = ["l3"]
# MEMS = ["l1", "l3", "mem"]
MEM_NAMES = {
    "l1": "Small",
    "l3": "Medium",
    "mem": "Large",
}

PLATFORMS = ["icelake", "sapphirerapids", "znver4"]

PLATFORM_NAMES = {
    "icelake": "Ice Lake",
    "sapphirerapids": "Em. Rapids",
    "znver4": "Zen 4",
}

LOWD_ALGORITHMS = [
    "bmiss", "bmiss_sttni",
    "qfilter",
    "shuffling_sse", "shuffling_avx2", "shuffling_avx512",
    "broadcast_sse", "broadcast_avx2", "broadcast_avx512",
    "vp2intersect_emulation",
]

HIGHD_ALGORITHMS = [
    "qfilter_bsr",
    "shuffling_sse_bsr", "shuffling_avx2_bsr", "shuffling_avx512_bsr",
    "broadcast_sse_bsr", "broadcast_avx2_bsr", "broadcast_avx512_bsr",
    # "croaring",
    # "croaring_inplace",
]

SKEW_ALGORITHMS = [
    "galloping_lut",
    "galloping_sse_lut", "galloping_avx2_lut", "galloping_avx512_lut",
    "lbk_v1x4_sse_lut",
    "lbk_v1x8_sse_lut",
    "lbk_v3_sse_lut",
    "lbk_v1x8_avx2_lut",
    "lbk_v1x16_avx2_lut",
    "lbk_v3_avx2_lut",
    "lbk_v3_avx512_lut",
    "lbk_v1x16_avx512_lut",
    "lbk_v1x32_avx512_lut",
]

VARY = "skew"
INDEX = "skewness_factor"

ALGORITHMS = LOWD_ALGORITHMS + HIGHD_ALGORITHMS + SKEW_ALGORITHMS

SPECIAL = ["croaring", "croaring_inplace"]

VARIANTS = ["lut", "comp", "br_lut", "br_comp"]
VARIANT_NAMES = {
    "lut": "Branchless Lookup",
    "comp": "Branchless Compress",
    "br_lut": "Branch Lookup",
    "br_comp": "Branch Compress",
}

LUTS = ["lut", "br_lut"]
COMPS = ["comp", "br_comp"]
BRANCHLESS = ["lut", "comp"]
BRANCH = ["br_lut", "br_comp"]


def main():
    nrows = len(MEMS)
    ncols = len(PLATFORMS)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(3 * ncols, 2.5 * nrows)
                            # figsize=(1.5 * ncols, 1 * nrows),
                            )
                            # sharex=True, sharey="row")

    unused_colors = sns.color_palette("tab10", 10)
    line_colors = {}

    for i, platform in enumerate(PLATFORMS):
        for j, mem in enumerate(MEMS):
            alg_data = get_algorithm_data(platform, mem)

            alg_times = pd.DataFrame()
            for alg, df in alg_data.items():
                df = df.set_index(INDEX)
                alg_times[alg] = df["time_ns"]
            
            best_algs = alg_times.idxmin(axis=1)
            print(f"{platform} {mem}: {list(dict.fromkeys(best_algs.values))}")

            alg_data = {alg: df for alg, df in alg_data.items()
                        if alg in best_algs.values or alg in SPECIAL}

            ax = axs[i]
            # ax = axs[j, i]

            for alg, df in alg_data.items():
                if alg in line_colors:
                    color = line_colors[alg]
                else:
                    if len(unused_colors) == 0:
                        unused_colors = sns.color_palette("tab10", 10)
                    color = unused_colors.pop(0)
                    line_colors[alg] = color

                # Put newline before first bracket
                name = label.algorithm_label(alg).replace(" (", "\n(", 1)

                ax.plot(df[INDEX], df["throughput_vs_branchless_merge_lut"],
                        label=name, color=color)
                
                if VARY == "density":
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                if i == 0:
                    ax.set_ylabel(f"{MEM_NAMES[mem]}")
                if j == nrows - 1:
                    ax.set_xlabel(f"{PLATFORM_NAMES[platform]}")

                if VARY == "density":
                    xticks = [0.001, 0.01, 0.1, 1]
                    ax.xaxis.set_ticks(xticks)
                    
                    ax.tick_params(axis='x', labelsize=8)
                    ax.tick_params(axis='y', labelsize=8)
                    plt.minorticks_off()
    
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    by_label = dict(zip(labels, lines))
    fig.legend(by_label.values(), by_label.keys(), ncols=1,
               bbox_to_anchor=(1.1, 0.45),
            #    bbox_to_anchor=(1.26, 0.42),
               loc="center right")

    # fig.suptitle("Best Low Density Algorithms per Environment", size="x-large")
    fig.supxlabel(VARY.title(), y=-0.1, size="medium")
    fig.supylabel("Relative Throughput (Merge=1)", x=-0.03, size="medium")

    plt.savefig(f"../../plots/best/best-{VARY}.pdf", bbox_inches="tight")
    plt.show()


def get_algorithm_data(platform, mem):
    alg_data = {}
    for alg in ALGORITHMS:
        for variant in VARIANTS:
            if "bmiss" in alg and variant in COMPS:
                continue

            alg_id = f"{alg}_{variant}"
            df = load_data(platform, mem, alg, VARY, variant)
            if df is not None:
                alg_data[alg_id] = df
    
    for alg in SKEW_ALGORITHMS:
        df = load_data(platform, mem, alg, VARY, None)
        if df is not None:
            alg_data[alg] = df

    # alg_data["croaring"] = load_data(platform, mem, "croaring", VARY, None)
    # alg_data["croaring_inplace"] = load_data(platform, mem, "croaring_inplace", VARY, None)

    return alg_data



def load_data(platform, mem, alg, vary, variant):
    if variant is not None:
        file = RESULTS / vary / platform / mem / f"{alg}_{variant}.csv"
    else:
        file = RESULTS / vary / platform / mem / f"{alg}.csv"
    if file.is_file():
        return pd.read_csv(file)
    else:
        return None


if __name__ == "__main__":
    main()

# %%

