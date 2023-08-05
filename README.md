# SIMD Set Operations
This library is split into two packages: **`setops`** containing various set
intersection implementations, and **`benchmark`** containing a number of
executables used to run experiments.

## Set intersection library (`setops/`)
This library contains implementations for a wide range of set intersection
algorithm, some scalar and some SIMD. Many SIMD algorithms are implemented using
Rust's portable [`simd`](https://doc.rust-lang.org/std/simd/index.html) library,
however some algorithms fall back on x86 intrinsics in cases where the library
does not provide the necessary instruction.

Set intersection algorithms can be classified by several characteristics:
- whether they operate on **two** sets or ***k*** sets,
- whether they are **scalar** or **vector** (i.e., use SIMD instructions),
- whether they operate on **sorted arrays** or some **custom** data structure.

### 2-set algorithms
So far, the following 2-set (pair) algorithms have been implemented. Please see
implementations for reference to original authors.

**Scalar**
- for sets of similar size: merge variants `naive_merge`, `branchless_merge`,
`bmiss_scalar_3x/4x`.
- for skewed intersection: `galloping`
- adaptive algorithm: `baezayates`

**Vector**
- shuffling variants: `shuffling_[sse, avx2, avx512]`
from [this blog](https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/),
found in [`simd_shuffling.rs`](setops/src/intersect/simd_shuffling.rs)
- broadcast variants: `broadcast_[sse, avx2, avx512]`,
found in [`simd_shuffling.rs`](setops/src/intersect/simd_shuffling.rs)
- galloping variants: `galloping_[sse, avx2, avx512]`,
from [this paper](https://arxiv.org/abs/1401.6399),
found in [`simd_galloping.rs`](setops/src/intersect/simd_galloping.rs)
- `bmiss` and `bmiss_sttni`
from [this paper](https://dl.acm.org/doi/10.14778/2735508.2735518),
found in [`bmiss.rs`](setops/src/intersect/bmiss.rs)
- `qfilter` from [this paper](https://dl.acm.org/doi/10.1145/3183713.3196924),
found in [`qfilter.rs`](setops/src/intersect/qfilter.rs)
- `fesia` from [this paper](https://ieeexplore.ieee.org/abstract/document/9101681),
found in [`fesia.rs`](setops/src/intersect/fesia.rs).
This algorithm uses a custom bitmap data structure.
- `vp2intersect_emulation` from [this paper](https://arxiv.org/pdf/2112.06342.pdf)
and `conflict_intersect` from [tetzank](https://github.com/tetzank/SIMDSetOperations)
can be found in [`avx512.rs`](setops/src/intersect/avx512.rs)

**BSR**
- [Base and State Representation](https://dl.acm.org/doi/abs/10.1145/3183713.3196924)
(BSR) is a custom bitmap representation designed for fast intersection of dense
datasets (aimed at graph applications). Many of the above algorithms have BSR
variants with `_bsr` appended to their names. This representation was intended
for use with the `qfilter` algorithm.


### k-set algorithms
- classical adaptive algorithms such as `adaptive`, `small_adaptive` and
`baezayates` can be found in [`adaptive.rs`](setops/src/intersect/adaptive.rs).
- all 2-set algorithms which operate on a sorted array of integers can be
extended to k-set with the function `svs_generic` (in
[`svs.rs`](setops/src/intersect/svs.rs))


## Benchmarking library (`benchmark/`)

The benchmark library consists of four [binary
targets](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#binaries)
: `generate`, `benchmark`, `plot` and `datatest`. The general workflow is to
first define experiments and datasets in `experiment.toml`, then to run
`generate` to generate the datasets, `benchmark` to run the experiments and
`plot` to plot the results. This separation allows benchmarks to be run without
regenerating the datasets. The tool `datatest` allows dataset characteristics to
be verified.

### Step 1: create `experiment.toml`
`experiment.toml` contains a list of datasets, algorithm sets and experiments.
First, define a dataset with `[[dataset]]`. Then, specify groups of algorithms
to be plotted together in the `[algorithm_sets]` table. Finally specify an
`[[experiment]]` to run set of algorithms on a specified dataset.

#### `[[dataset]]`
A dataset consists of sequence of x-values each containing `gen_count` groups of
sets. The parameter to be varied over the x-axis is defined by `vary`. If
`vary = "selectivity"`, selectivity will be varied from `selectivity` to `to`
with a step of `step`. For a given x-value, the $i$ th group is written to the
datafile found at `datasets/<id>/<x>/<i>`.

The following parameters are defined on an intersection group of `set_count = k`
sets, $S_1\cap S_2\cap ...\cap S_k$, where $S_1$ is the largest set and $S_k$ is
the smallest set.

- `density` defines the ratio of the size of the largest set to the size of the
element space `m`. The space of elements is always `{0,1,...,m-1}`. For
example, if the size of the largest set is 16 and the density is 0.25, the
element space is `{0,1,...,63}`. Density is represented with an integer from 0
to 1000 mapping to a density of 0 to 100%.

- `selectivity` defines the ratio of the size of an intersection's output to the
size of the smallest input set. It is represented with an integer from `0` to
`1000` mapping to a selectivity of 0 to 100%.

- `max_len` defines the size (cardinality) of the largest set. `size = n`
results in an actual set size of $2^n$.

- `skewness_factor` defines the size of sets in relation to the size of the
largest set. It is represented by an integer $s$ which maps to the floating
point number $f=s/1000.0$. The size of the $k$ th set with respect to the
largest set is $ |S_k| = |S_1|/k^f $. This ensures set sizes are inversely
proportional to their rank $k$
(see [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law)).


The following example illustrates how to generate a pairwise intersection with
varying selectivity.
```toml
[[dataset]]
name = "2set_vary_selectivity" # unique id
set_count = 2         # pairwise intersection
gen_count = 10        # generate 10 pairs
vary = "selectivity"  # vary selectivity along x-axis
selectivity = 0       # from 0-100% with a step of 10%
to = 1000
step = 100
skewness_factor = 0   # skew of 1:2^0 === 1:1
density = 1           # density 0.1%
max_len = 20          # each 2^20 (approx 1M) elements
```

> Note 1: it is possible to specify `selectivity` and `density` parameters which
are unattainable together. Run `datatest` to verify intersection groups match
parameters. The generator will prioritize density over selectivity, so the
selectivity will increase if the density is too high.

> Note 2: for $k$-set benchmarks where $k\ge 3$, the `selectivity` specified is
a *minimum* value. It is possible for the selectivity to increase slightly if
the random number generator happens to add the same element in all sets. Use
`datatest` to get an accurate measure of this variance. Synthetic $k$-set
generation may not be realistic as elements are likely to appear in either very
few or all generated sets. This issue is not present for 2-set datasets. 

#### `[algorithm_sets]` and `[[experiment]]`
An *experiment* is a set of *algorithms* benchmarked on a specific *dataset*.
To define the set of algorithms to be included, specify them in the
`algorithm_set` table as shown below. Many `experiment`s may share the same
`algorithm_set`.

Multiple experiments may also share a single dataset. If such experiments share
algorithms, each algorithm will benchmarked once and the results for these
algorithms will appear in both `experiment` plots. An example experiment
definition is shown below.
```toml
[algorithm_sets]
scalar_2set = [
    "naive_merge", "branchless_merge",
    "bmiss_scalar_3x", "bmiss_scalar_4x",
]
# ...

[[experiment]]
name = "scalar_2set_vary_selectivity"
title = "Scalar 2-set varying selectivity"
dataset = "2set_vary_selectivity"
algorithm_set = "scalar_2set"
```
See [`benchmark.rs`](benchmark/src/bin/benchmark.rs) for a list of algorithms.

### Step 2: run `generate`
To build datasets, run the generator with
```sh
cargo run --release --bin=generate
```

### Step 3: run `benchmark`
After running the following command, results can be found in `results.json`.
```sh
cargo run --release --bin=benchmark
```

### Step 4: run `plot`
```sh
cargo run --release --bin=plot
```

> Run these programs with `--help` for info about additional arguments.

### Verifying datasets with `datatest`
A fourth, optional program `datatest` validates datasets and outputs a warning
if any dataset parameters vary more than a given threshold. Users are encouraged
to view and edit `benchmark/src/bin/datatest.rs` to understand and tweak
thresholds (or just output everything).
```sh
cargo run --release --bin=datatest
```

*This library was developed for Alex Brown's honours thesis at UNSW*
