#!/usr/bin/env python3
# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

mpl.rcParams['figure.dpi'] = 300

RESULTS = Path("../../processed/compare/")

MEMS = ["l1", "l3", "mem"]
MEM_NAMES = {
    "l1": "Small",
    "l3": "Medium",
    "mem": "Large",
}

PLATFORMS = ["sapphirerapids", "znver4"]
# PLATFORMS = ["icelake", "sapphirerapids", "znver4"]

PLATFORM_NAMES = {
    "icelake": "IL",
    "sapphirerapids": "ER",
    "znver4": "Zn4",
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
    "bmiss_sttni": "BMissSTTNI",
    "qfilter": "QFilter",
    "shuffling_sse": "Shuffle128",
    "shuffling_avx2": "Shuffle256",
    "shuffling_avx512": "Shuffle512",
    "broadcast_sse": "Bcast128",
    "broadcast_avx2": "Bcast256",
    "broadcast_avx512": "Bcast512",
    "vp2intersect_emulation": "VP2Emul",
}

HIGHD_ALGORITHMS = [
    "qfilter_bsr",
    "shuffling_sse_bsr", "shuffling_avx2_bsr", "shuffling_avx512_bsr",
    "broadcast_sse_bsr", "broadcast_avx2_bsr", "broadcast_avx512_bsr",
]

HIGHD_NAMES = {
    "qfilter_bsr": "QFilterBSR",
    "shuffling_sse_bsr": "Shuffle128BSR",
    "shuffling_avx2_bsr": "Shuffle256BSR",
    "shuffling_avx512_bsr": "Shuffle512BSR",
    "broadcast_sse_bsr": "Bcast128BSR",
    "broadcast_avx2_bsr": "Bcast256BSR",
    "broadcast_avx512_bsr": "Bcast512BSR",
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
    plot_bars()
    plot_heat()

def plot_bars():
    nrows = len(MEMS) * len(PLATFORMS)
    ncols = len(CATEGORIES)

    width_ratios = [len(CATEGORY_GROUPS[cat]) for cat in CATEGORIES]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 0.8*nrows),
                            sharex="col", sharey="row", width_ratios=width_ratios)
    fig.subplots_adjust(hspace=0.5, top=0.9)

    fig.suptitle("Performance of Common Optimisations")

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
                        ylabel=f"{PLATFORM_NAMES[platform]}, {MEM_NAMES[mem]}")

                ax.yaxis.set_label_coords(5.0, 1.05)
                # rotate yaxis label to 0
                ax.yaxis.label.set_rotation(0)

                if i == 0 and j == 0 and k == 0:
                    handles, labels = ax.get_legend_handles_labels()

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
    plt.ylabel("Relative Throughput (slowest=0, fastest=1)", )

    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

    # Save to disk
    plt.savefig("../../plots/compare-out/bars.pdf", bbox_inches="tight")
    plt.show()


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
    throughputs = load_data(platform, mem, alg, vary)
    
    smallest = min(throughputs, key=throughputs.get)
    largest = max(throughputs, key=throughputs.get)

    # linearly interpolate between smallest and largest (0 - 1)
    scaled = {k: (v - throughputs[smallest]) / (throughputs[largest] - throughputs[smallest])
              for k, v in throughputs.items()}

    return scaled

def plot_heat():
    plat_mems = [(p, m) for p in PLATFORMS for m in MEMS]

    plat_mem_strs = [f"{MEM_NAMES[m]} sets, {PLATFORM_NAMES[p]}" for p, m in plat_mems]

    df_branch = pd.DataFrame(index=plat_mem_strs)
    df_comp = pd.DataFrame(index=plat_mem_strs)

    def add(algs, vary):
        for algorithm in algs:
            branch_col = []
            comp_col = []
            for platform, mem in plat_mems:
                branch_tendency, comp_tendency = gen_heat(platform, mem, algorithm, vary)

                branch_col.append(branch_tendency)
                comp_col.append(comp_tendency)
            
            alg_name = LOWD_NAMES.get(algorithm) if algorithm in LOWD_ALGORITHMS else HIGHD_NAMES.get(algorithm)
            df_branch[alg_name] = branch_col
            df_comp[alg_name] = comp_col
    
    add(LOWD_ALGORITHMS, "selectivity")
    add(HIGHD_ALGORITHMS, "density")

    size = (6, 3)
    fig, ax = plt.subplots(figsize=size)

    cbar_kws={"shrink": 0.9,
              "pad": 0.04,
              "anchor": (0, -1.3)}
    
    SPACES = 20

    sns.heatmap(ax=ax, data=df_branch, square=True, cmap="vlag",
                cbar_kws={"label": "branchless" + " "*SPACES + "branch",
                          **cbar_kws})
    # ax.set_title("Factor to which Branching Improves Throughput")
    plt.xticks(rotation=90)

    plt.savefig("../../plots/compare-out/heat-branching.pdf", bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(ax=ax, data=df_comp, square=True, cmap="vlag",
                cbar_kws={"label": "lookup" + " "*SPACES + "compressd",
                          **cbar_kws})
    # ax.set_title("Factor to which COMPRESSD Improves Throughput")
    plt.xticks(rotation=90)

    plt.savefig("../../plots/compare-out/heat-compress.pdf", bbox_inches="tight")
    plt.show()



def gen_heat(platform, mem, alg, vary):
    throughputs = load_data(platform, mem, alg, vary)

    smallest = min(throughputs, key=throughputs.get)
    largest = max(throughputs, key=throughputs.get)

    branchless_avg = (throughputs["lut"] + throughputs["comp"]) / 2
    branch_avg = (throughputs["br_lut"] + throughputs["br_comp"]) / 2

    lut_avg = (throughputs["lut"] + throughputs["br_lut"]) / 2
    comp_avg = (throughputs["comp"] + throughputs["br_comp"]) / 2

    branch_tendency = (branch_avg - branchless_avg) / (throughputs[largest] - throughputs[smallest])
    comp_tendency = (comp_avg - lut_avg) / (throughputs[largest] - throughputs[smallest])

    return branch_tendency, comp_tendency


def load_data(platform, mem, alg, vary):

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
    
    return throughputs

if __name__ == "__main__":
    main()

# %%
