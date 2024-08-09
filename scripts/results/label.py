import re

CACHE_NAMES = {
    "l1d": "L1 Data",
    "l1i": "L1 Instr.",
    "ll": "Last Level",
}
CACHE_OP = {
    "rd": "Read",
    "wr": "Write",
}
CACHE_EVENT = {
    "access": "Accesses",
    "miss": "Misses",
    "miss_rate": "Miss rate",
}

COL_TITLES = {
    "selectivity": "Selectivity",
    "density": "Density",
    "size": "Size",
    "skewness_factor": "Skew",
    "set_count": "Set Count",
    "element_count": "Element Count",
    "element_bytes": "Total size of all sets (bytes)",
    "element_bytes_pow": "Total size of all sets (bytes)",
    "throughput_eps": "Throughput (elements/s)",
    "throughput/density": "Throughput / density",
    "time_s": "Intersection Time (s)",
    "time_s/element": "Time per Element (s)",
    "time_ns": "Intersection Time (ns)",
    "time_ns/element": "Time per Element (ns)",
    "branches": "Total Branches",
    "branches/element": "Branches per Element",
    "branch_misses": "Total Branch Misses",
    "branch_miss_rate": "Branch Miss Rate",
    "branch_misses/element": "Branch Misses per Element",
    "instructions": "Total Retired Instructions",
    "instructions/element": "Retired Instructions per Element",
    "cpi": "Cycles per Instruction",
    "ipc": "Instructions per Clock",
    "cpu_cycles": "Total CPU Cycles",
    "cpu_cycles/element": "CPU Cycles per Element",
    "cpu_cycles_ref": "Total CPU Cycles (adjusted for frequency scaling)",
    "cpu_cycles_ref/element": "CPU Cycles per Element (adjusted for frequency scaling)",
}

SCALAR_ALGORITHMS = {
    "naive_merge": "Merge (branch)",
    "branchless_merge": "Merge (branchless)",
}

VECTOR_ALGORITHMS = {
    "qfilter": "QFilter",
    "qfilter_c": "QFilter(FFI)",
    "bmiss": "BMiss",
    "bmiss_sttni": "BMissSTTNI",
    "shuffling_sse": "Shuffle128",
    "shuffling_avx2": "Shuffle256",
    "shuffling_avx512": "Shuffle512",
    "broadcast_sse": "Bcast128",
    "broadcast_avx2": "Bcast256",
    "broadcast_avx512": "Bcast512",
    "vp2intersect_emulation": "VP2Emul",
    "croaring": "Roaring",
    # TODO
    "lbk_v3_sse": "LBK v3 128",
    "lbk_v3_avx2": "LBK v3 256",
    "lbk_v3_avx512": "LBK v3 512",
    "lbk_v1x4_sse": "LBK v1 x4 128",
    "lbk_v1x8_sse": "LBK v1 x8 128",
    "lbk_v1x8_avx2": "LBK v1 x8 256",
    "lbk_v1x16_avx2": "LBK v1 x16 256",
    "lbk_v1x16_avx512": "LBK v1 x16 512",
    "lbk_v1x32_avx512": "LBK v1 x32 512",
    "galloping": "Galloping",
    "galloping_sse": "Gallop128",
    "galloping_avx2": "Gallop256",
    "galloping_avx512": "Gallop512",
}

SAVE_TYPE = {
    "_count": "COUNT",
    "_lut": "LUT",
    "_comp": "COMP",
}

NO_SAVE = [
    "merge",
    "bmiss",
    "galloping",
    "lbk",
    "croaring",
]

NO_BRANCH = [
    "merge",
    "galloping",
    "lbk",
    "croaring",
]


def col_title(col): 
    title = COL_TITLES.get(col)
    if title is not None:
        return title
    else:
        relative = re.match(r"^throughput_vs_(.*)$", col)
        if relative:
            return f"Relative throughput ({relative.group(1)}=1)"
        
        cache = re.match(r"^(l1d|l1i|ll)_(rd|wr)_(access|miss|miss_rate)(/element)?$", col)
        if cache:
            return f"{CACHE_NAMES[cache.group(1)]} {CACHE_OP[cache.group(2)]} {CACHE_EVENT[cache.group(3)]}{ ' per element' if cache.group(4) else ''}"

        print(f"Unknown col: {col}")
        return None

def col_formatter(col):
    if col == "element_bytes":
        return lambda x: format_unit(x, 1000, ['B', 'KB', 'MB', 'GB', 'TB'])
    if col == "element_bytes_pow":
        return lambda x: format_unit_pow(x, 10, ['B', 'KiB', 'MiB', 'GiB', 'TiB'])
    elif col == "time_ns":
        return lambda x: format_unit(x, 1000, ['ns', 'us', 'ms', 's'])
    else:
        return None

def format_unit(value, power, labels):
    n = 0
    while value > power and n < len(labels) - 1:
        value /= power
        n += 1
    return "{:.0f}".format(value) + labels[n]

def format_unit_pow(value, exp, labels):
    unit = int(value) // exp
    offset = int(value) % exp
    return f"{1 << offset}{labels[unit]}"

def do_log(col):
    return col in ["element_count", "element_bytes"]

def algorithm_label(alg, label="", has_branch=False):

    scalar = SCALAR_ALGORITHMS.get(alg)
    if scalar is not None:
        return scalar + label

    vector = VECTOR_ALGORITHMS.get(alg)
    if vector is not None:
        if not has_branch and not any(x in alg for x in NO_BRANCH):
            label = " (!BR)" + label
        return vector + label
    
    for postfix, tag in SAVE_TYPE.items():
        if alg.endswith(postfix):
            rest = alg[:-len(postfix)]
            tagged = f" ({tag})" + label if not any(x in alg for x in NO_SAVE) else label
            return algorithm_label(rest, tagged, has_branch) 
    
    if alg.endswith("_br"):
        return algorithm_label(alg[:-3], " (BR)" + label, True) 
    
    if alg.endswith("_bsr"):
        if not has_branch and not any(x in alg for x in NO_BRANCH):
            label = " (branchless)" + label
        return algorithm_label(alg[:-4], " BSR" + label, True)
            
    return f"FAIL: {alg}{label}"
