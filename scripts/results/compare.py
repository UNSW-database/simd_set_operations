#!/usr/bin/env python3
# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("../../processed/compare/")

MEMS = ["l1", "l3", "mem"]
MEM_NAMES = {
    "l1": "L1D cache",
    "l3": "L3 cache",
    "mem": "main mem.",
}

PLATFORMS = ["sapphirerapids", "znver4"]

PLATFORM_NAMES = {
    "sapphirerapids": "Sapph. Rapids",
    "znver4": "Zen 4",
}

LOWD_ALGORITHMS = [
    "bmiss", "bmiss_sttni",
    "qfilter",
    "shuffling_sse", "shuffling_avx2", "shuffling_avx512",
    "broadcast_sse", "broadcast_avx2", "broadcast_avx512",
    "vp2intersect_emulation",
]

LOWD_NAMES = {
    "bmiss": "BMiss",
    "bmiss_sttni": "BMiss\nSTTNI",
    "qfilter": "QFilter",
    "shuffling_sse": "Shuffling\nSSE",
    "shuffling_avx2": "Shuffling\nAVX2",
    "shuffling_avx512": "Shuffling\nAVX512",
    "broadcast_sse": "Broadcast\nSSE",
    "broadcast_avx2": "Broadcast\nAVX2",
    "broadcast_avx512": "Broadcast\nAVX512",
    "vp2intersect_emulation": "VP2INT.\nEmul.",
}

HIGHD_ALGORITHMS = [
    "qfilter_bsr",
    "broadcast_sse_bsr", "broadcast_avx2_bsr", "broadcast_avx512_bsr",
    "shuffling_sse_bsr", "shuffling_avx2_bsr", "shuffling_avx512_bsr",
]

HIGHD_NAMES = {
    "broadcast_sse_bsr": "Broadcast\nSSE BSR",
    "broadcast_avx2_bsr": "Broadcast\nAVX2 BSR",
    "broadcast_avx512_bsr": "Broadcast\nAVX512 BSR",
    "qfilter_bsr": "QFilter\nBSR",
    "shuffling_sse_bsr": "Shuffling\nSSE BSR",
    "shuffling_avx2_bsr": "Shuffling\nAVX2 BSR",
    "shuffling_avx512_bsr": "Shuffling\nAVX512 BSR",
}

VARIANTS = ["lut", "comp", "br_lut", "br_comp"]
VARIANT_NAMES = {
    "lut": "Branchless Lookup",
    "comp": "Branchless Compress",
    "br_lut": "Branch Lookup",
    "br_comp": "Branch Compress",

}

VARY_LIMIT = 0.5

LUTS = ["lut", "br_lut"]
COMPS = ["comp", "br_comp"]
BRANCHLESS = ["lut", "comp"]
BRANCH = ["br_lut", "br_comp"]

CATEGORIES = [
    "bmiss", "qfilter", "shuffling", "broadcast", "vp2intersect_emulation",
    "qfilter_bsr", "shuffling_bsr", "broadcast_bsr",
]
CATEGORY_GROUPS = {
    "bmiss": ["bmiss", "bmiss_sttni"],
    "qfilter": ["qfilter"],
    "shuffling": ["shuffling_sse", "shuffling_avx2", "shuffling_avx512"],
    "broadcast": ["broadcast_sse", "broadcast_avx2", "broadcast_avx512"],
    "vp2intersect_emulation": ["vp2intersect_emulation"],
    "qfilter_bsr": ["qfilter_bsr"],
    "broadcast_bsr": ["broadcast_sse_bsr", "broadcast_avx2_bsr", "broadcast_avx512_bsr"],
    "shuffling_bsr": ["shuffling_sse_bsr", "shuffling_avx2_bsr", "shuffling_avx512_bsr"],
}
CATEGORY_NAMES = {
    "bmiss": "BMiss",
    "qfilter": "QFilter",
    "shuffling": "Shuffling",
    "broadcast": "Broadcast",
    "vp2intersect_emulation": "Emul.",
    "qfilter_bsr": "BSR",
    "broadcast_bsr": "Broadcast BSR",
    "shuffling_bsr": "Shuffling BSR",
}

WIDTHS = {
    "bmiss": "Base",
    "bmiss_sttni": "STTNI",
    "qfilter": "",
    "shuffling_sse": "SSE",
    "shuffling_avx2": "AVX2",
    "shuffling_avx512": "AVX512",
    "broadcast_sse": "SSE",
    "broadcast_avx2": "AVX2",
    "broadcast_avx512": "AVX512",
    "vp2intersect_emulation": "VP2INT.",
    "qfilter_bsr": "QFilter",
    "broadcast_sse_bsr": "SSE",
    "broadcast_avx2_bsr": "AVX2",
    "broadcast_avx512_bsr": "AVX512",
    "shuffling_sse_bsr": "SSE",
    "shuffling_avx2_bsr": "AVX2",
    "shuffling_avx512_bsr": "AVX512",
}


def main():

    nrows = len(MEMS) * len(PLATFORMS)
    ncols = len(CATEGORIES)

    width_ratios = [len(CATEGORY_GROUPS[cat]) for cat in CATEGORIES]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 1*nrows),
                            sharex="col", sharey="row", width_ratios=width_ratios,)
    fig.subplots_adjust(hspace=0.5, top=0.9)

    fig.suptitle("Performance of common optimisations")

    for i, platform in enumerate(PLATFORMS):
        for j, mem in enumerate(MEMS):
            df_original = process_data(platform, mem)

            for k, category in enumerate(CATEGORIES):
                ax = axs[i*len(MEMS) + j, k]

                df = df_original.copy()
                # Filter rows by category
                df = df.loc[CATEGORY_GROUPS[category]]

                df.index = [f"{WIDTHS[alg]}" for alg in df.index]
                df.plot(ax=ax, kind="bar", legend=False, rot=0,
                        xlabel=CATEGORY_NAMES[category],
                        ylabel=f"{PLATFORM_NAMES[platform]} {MEM_NAMES[mem]}")

                ax.yaxis.set_label_coords(5.0, 1.05)
                # rotate yaxis label to 0
                ax.yaxis.label.set_rotation(0)

                if i == 0 and j == 0 and k == 0:
                    handles, labels = ax.get_legend_handles_labels()



    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
    plt.ylabel("Relative Throughput (slowest=0, fastest=1)", )

    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))


def process_data(platform, mem):
    lowd_data = {k: gen_scaled(platform, mem, k, "selectivity") for k in LOWD_ALGORITHMS}
    highd_data = {k: gen_scaled(platform, mem, k, "density") for k in HIGHD_ALGORITHMS}

    algorithms = LOWD_ALGORITHMS + HIGHD_ALGORITHMS

    all_data = {**lowd_data, **highd_data}

    df = pd.DataFrame(index=algorithms)
    for variant in VARIANTS:
        df[VARIANT_NAMES[variant]] = [all_data[alg][variant] for alg in algorithms]

    return df


def gen_scaled(platform, mem, alg, vary):

    dlabel = "lowd" if alg in LOWD_ALGORITHMS else "highd"
    dir = RESULTS / platform / f"compare-{dlabel}-{mem}" / f"compare_{alg}_{vary}_{mem}"

    data = {}
    for variant in VARIANTS:
        df = pd.read_csv(dir / f"{alg}_{variant}.csv")
        df = df.loc[df[vary] <= 0.5]
        data[variant] = df

    throughputs = {}
    for variant in VARIANTS:
        throughputs[variant] = data[variant]["throughput_eps"].mean()
    
    smallest = min(throughputs, key=throughputs.get)
    largest = max(throughputs, key=throughputs.get)

    # linearly interpolate between smallest and largest (0 - 1)
    scaled = {k: (v - throughputs[smallest]) / (throughputs[largest] - throughputs[smallest])
              for k, v in throughputs.items()}

    return scaled


if __name__ == "__main__":
    main()

# %%
