# SIMD Set Operations
This library is split into two packages: a library **`setops`** containing
various set intersection implementations, and **`benchmark`** containing a
number of executables used to run experiments.

## Set intersection library (`setops/`)
This library contains a benchmark suite for a wide range of set intersection
algorithm. These algorithms can be classified by several characteristics.
- whether they operate on a **pair** of sets or ***k*** sets,
- whether they are **scalar** or **vector** (i.e., use SIMD instructions),
- whether they operate on **sorted arrays** or some **alternative
  datastructure**.


## Benchmarking library (`benchmark/`)
The benchmark library consists of three [binary
targets](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#binaries)
: `generate`, `benchmark` and `plot`. The general workflow is to first define
experiments and datasets in `experiment.toml`, then to run `generate` to
generate the datasets, `benchmark` to run the experiments and `plot` to plot the
results. This separation allows benchmarks to be run without regenerating the
datasets.

### step 1: create `experiment.toml`
`experiment.toml` contains a list of datasets and experiments. A dataset may
vary one of the below parameters, and all others are fixed. The generator will
generate `count` pairs/groups of sets to be used in benchmarking.

There are four parameters than can be set for a pair of sets: `density`,
`selectivity`, `skew` and `size`.

- `density` defines the ratio of the size of the largest set to the size of the
element space $m$. The space of elements is always $\{0, 1, ..., m-1\}$. For
example, if the size of the largest set is 16 and the density is 0.25, the
element space is $\{0,1,...,63\}$. `density` is represented with an integer from
0 to 1000 mapping to a density of 0 to 100%.

- `selectivity` defines the ratio of the size of an intersection's output to the
size of the smallest input set. It is represented with an integer from `0` to
`1000` mapping to a selectivity of 0 to 100%.

- `skew` defines the ratio of the large set to the small set. It is represented
by an integer $s$, where the ratio of set sizes is $2^{s-1}:1$. This means a
skew of `1` results in a size ratio of $1:1$.

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
Multiple `experiment`s can be defined on a single `dataset`. If such experiments
share algorithms, each algorithm will benchmarked once and the results for these
algorithms will appear in both `experiment` plots. An example is shown below.
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
