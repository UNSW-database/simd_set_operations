
ALGS = [
    "shuffling_sse",
    "shuffling_avx2",
    "shuffling_avx512",
    "broadcast_sse",
    "broadcast_avx2",
    "broadcast_avx512",
    "bmiss",
    "bmiss_sttni",
    "qfilter",
    "vp2intersect_emulation",
]

for alg in ALGS:
    variants = []
    for br in ["", "_br"]:
        for save in ["_lut", "_comp"]:
            variants.append(f"{alg}{br}{save}")
    
    variants_str = ", ".join([f'"{v}"' for v in variants])
    print(f"compare_{alg} = [ {variants_str} ]")

def gen_exp(alg, vary):
    return f'''[[experiment]]
name = "compare_{alg}"
dataset = "2set_vary_{vary}"
algorithm_set = "compare_{alg}"
    '''

for alg in ALGS:
    print(gen_exp(alg, "size"))
    print(gen_exp(alg, "selectivity"))
