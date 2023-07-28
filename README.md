# SIMD Set Operations
This library is split into two packages: a library **`setops`** containing
various set intersection implementations, and **`benchmark`** containing a
number of executables used to run experiments.

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
found in [`shuffling.rs`](setops/src/intersect/simd_shuffling.rs)
- broadcast variants: `broadcast_[sse, avx2, avx512]`,
found in [`shuffling.rs`](setops/src/intersect/simd_shuffling.rs)
- `bmiss` and `bmiss_sttni`
from [this paper](https://dl.acm.org/doi/10.14778/2735508.2735518),
found in [`bmiss.rs`](setops/src/intersect/bmiss.rs)
- `qfilter` from [this paper](https://dl.acm.org/doi/10.1145/3183713.3196924),
found in [`qfilter.rs`](setops/src/intersect/qfilter.rs)
- `fesia` from [this paper](https://ieeexplore.ieee.org/abstract/document/9101681),
found in [`fesia.rs`](setops/src/intersect/fesia.rs).
This algorithm uses a custom bitmap data structure.


**BSR**
- [Base and State Representation](https://dl.acm.org/doi/abs/10.1145/3183713.3196924)
(BSR) is a custom bitmap representation designed for fast intersection of dense
datasets (aimed at graph applications). Many of the above algorithms have BSR
variants with `_bsr` appended to the names. This representation was intended
for use with the `qfilter` algorithm.


### k-set algorithms
> TODO


## Benchmarking library (`benchmark/`)
> Currently only 2-set benchmarks are supported.

The benchmark library consists of three [binary
targets](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#binaries)
: `generate`, `benchmark` and `plot`. The general workflow is to first define
experiments and datasets in `experiment.toml`, then to run `generate` to
generate the datasets, `benchmark` to run the experiments and `plot` to plot the
results. This separation allows benchmarks to be run without regenerating the
datasets.

### Step 1: create `experiment.toml`
`experiment.toml` contains a list of datasets and experiments. A dataset may
vary one of the below parameters, and all others are fixed. The generator will
generate `count` pairs/groups of sets to be used in benchmarking.

There are four parameters than can be defined on pair of sets: `density`,
`selectivity`, `skew` and `size`.

- `density` defines the ratio of the size of the largest set to the size of the
element space `m`. The space of elements is always `{0,1,...,m-1}`. For
example, if the size of the largest set is 16 and the density is 0.25, the
element space is `{0,1,...,63}`. Density is represented with an integer from 0
to 1000 mapping to a density of 0 to 100%.

- `selectivity` defines the ratio of the size of an intersection's output to the
size of the smallest input set. It is represented with an integer from `0` to
`1000` mapping to a selectivity of 0 to 100%.

- `skew` defines the ratio of the large set's size to the small set's size. It
is represented by an integer $s$, where the ratio of set sizes is $2^{s-1}:1$.
This means a skew of `1` results in a size ratio of $1:1$, a skew of `3` results
in a ratio of $4:1$, and so on.

- `size` defines the cardinality of the smaller of the two sets. `size = n`
results in an actual set size of $2^n$. The `skew` value determines the size of
the large set. *Make sure this number is small if testing high skews*.

An example is shown
below.
```toml
[[dataset]]
name = "2set_vary_selectivity" # unique id
type = "2set"          # 2set or kset
vary = "selectivity"   # vary selectivity along x-axis
selectivity = 0        # from 0-100%, with a step of 10%.
to = 1000
step = 100
skew = 1               # fixed skew of 1:1
density = 1            # fixed density of 0.1%
size = 20              # fixed set size of 2^20 (~1M)
count = 10             # generate 10 pairs for each x-value.
# assumed to be uniformly distributed
```

An `experiment` is a set of algorithms benchmarked on a specific `dataset`.
Multiple experiments may use a single dataset. If such experiments share
algorithms, each algorithm will benchmarked once and the results for these
algorithms will appear in both `experiment` plots. An example experiment
definition is shown below.
```toml
[[experiment]]
name = "scalar_2set_vary_selectivity"
title = "Scalar 2-set varying selectivity"
dataset = "2set_vary_selectivity"
algorithms = [
    "naive_merge", "branchless_merge",
    "bmiss_scalar_3x", "bmiss_scalar_4x",
    "baezayates", "baezayates_opt",
]
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

*This library was developed for Alex Brown's honours thesis at UNSW*
