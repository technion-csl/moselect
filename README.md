# Introduction
Virtual-memory research increasingly relies on model-based evaluation: a partial simulator produces address-translation metrics, and an empirical runtime model maps those metrics to predicted benchmark runtime. This is far cheaper than full-system cycle-accurate simulation, but its accuracy depends on the training dataset used to build the model.

The key challenge is memory-layout selection. Each layout is a different mix of base pages and hugepages, and each benchmark execution under that layout contributes one training sample. Because the number of possible layouts grows exponentially with address-space size, exhaustive exploration is infeasible. The practical goal is therefore to select a small layout set that yields informative, well-distributed samples across the model input domain.

This repository provides an automated workflow for that problem, centered on Moselect: a coverage-driven layout selector that combines Intel PEBS profiling with execution feedback to fill uncovered intervals in the selector domain. In addition to Moselect, the repo includes Bayesian, genetic, and heuristic selectors, plus the surrounding infrastructure to run experiments, collect measurements, and build/analyze runtime models on x86-64 Linux systems.

# Quick Start
Simply clone this repo, enter the repo directory, and run `make`.
This will produce Mosmodel for the toy benchmark (random access over a 1GB array, takes ~15 seconds) provided in the repo.

# Software Prerequisites
- **Sudo permissions**: multiple steps require sudo priviliges, most notably, reserving hugepages and installing apt packages (perf, numactl, ...). We recommend to [configure sudo permissions without password](https://www.cyberciti.biz/faq/linux-unix-running-sudo-command-without-a-password/) to fully automate the workflow. Sudo with password may stop the workflow at these steps prompting for your password.
- **Linux distro**: The code was tested on Ubuntu 20 LTS. Please note that all necessary apt packages are downloaded automatically. If you are using a different Linux distribution, you should probably modify the makefile to use the proper package management software and package names.
- **Python**: Our scripts are written in Python3 and rely on python packages like numpy, pandas, matplotlib, scipy, scikit-optimize, bitarray, and scikit-learn (see `requirements.txt`). Run `make install-prereqs` to automatically install the required system packages (`python3`, `python3-venv`, `python3-pip`) and set up an isolated Python virtual environment at `.venv` with every package from `requirements.txt` (see `scripts/install_prereqs.sh`); then activate it with `source .venv/bin/activate` before running any experiments. Re-running `make install-prereqs` is safe/idempotent -- it reuses the existing `.venv` if one is already present. Alternatively, you may use your own Python3 environment (e.g., [Anaconda](https://www.anaconda.com/products/individual)) as long as it has the packages listed in `requirements.txt` installed.

# Hardware Prerequisites
- **Intel CPUs**: Moselect collects and analyzes the hardware performance counters of Intel CPUs for building Mosmodel. Additionally, our code uses Intel Precise Event Based Sampling (PEBS) to find interesting Mosalloc layouts. We successfully tested our code on Intel Broadwell processors or newer.

# Setup and Configuration
Before you start building and running Moselect, you need to set and configure the following:
- Update the following variables in `benchmark.mk`:
    - `BENCHMARK_PATH`: the full path to a directory containing the benchmark. This directory must contain the following files:
        - `pre_run.sh` - a script running before the "actual" benchmark is measured.
        - `run.sh` - the main script of the benchmark which will be measured with perf.
        - `post_run.sh` - a script running after the "actual" benchmark is measured.

# Moselect Directory Structure
- `mosalloc` - a git submodule pointing to the Mosalloc memory allocator.
- `scripts` - python scripts to run the experiments, collect the results, build Mosmodel and everything in between.
- `experiments` - every experiment (== a single run of the benchmark) is stored under this directory.
- `analysis` - CSV files with raw data and the model coefficients.
- `toy_benchmark` - a small-memory benchmark supplied with this repo to quickly demonstrate how Mosmodel is built using Moselect. It allocates a 1GB array and reads it randomly.
- `client_server_example` - a demo of how to create a benchmark infrastructure (`pre_run.sh`, `run.sh`, `post_run.sh`) for a client-server workloads, e.g., memcached.

# Artifact Testing (`make test_artifact`)
To run the artifact smoke test, execute:

```
make test_artifact
```

This target currently resolves to `short_moselect_test` in the top-level Makefile and does the following:

1. Runs `make install-prereqs` to provision required packages and the `.venv` environment.
2. Activates `.venv`.
3. Runs a short Moselect experiment with reduced cost:
    - `MOSELECT_NUM_OF_REPEATS=1`
    - `MOSELECT_NUM_LAYOUTS=25`
    - `MOSELECT_MAX_GAP=8`
4. Builds the Moselect analysis output via the `moselect` target (`analysis/moselect/scatter.pdf`).
5. After the smoke test completes, the measured points are plotted into `analysis/moselect/scatter.pdf`, so the generated layout/measurement cloud can be inspected as the artifact result.

In other words, `make test_artifact` is a lightweight artifact-validation path that executes the full Moselect pipeline (layout generation, benchmark runs, result collection, and analysis plotting) with a smaller budget and single repeat to shorten runtime.

For a short Bayesian-only smoke test, you can run:

```
make short_bayesian_test
```

# Layout Selectors
This repository includes four layout selectors, each exploring the hugepage-layout space with a different strategy.

## Moselect
Moselect is the main coverage-driven selector in this project. Unlike optimizers that focus on a single best point, Moselect tries to generate a compact set of layouts whose measured points are well distributed across the runtime/performance domain so that model fitting is stable and robust. In other words, the goal is not only to find a low-miss layout, but to cover the interesting parts of the hugepage-layout space with enough diversity that the downstream runtime model is well-conditioned.

### Problem formulation
The layout search space is combinatorial: each 2MB-aligned region of the `brk` heap can either use a base page or a hugepage, and the number of possible assignments grows exponentially with heap size. Exhaustive enumeration is therefore infeasible. Moselect instead treats the problem as a coverage task: it repeatedly chooses layouts that maximize the information gained about the unexplored regions of the domain, while keeping the total number of expensive benchmark runs small.

### How it works
1. **Bootstrap anchors.** It starts from PEBS-guided anchor layouts, including the trivial extremes (no hugepages and all hugepages), and augments them with a small set of regions that appear to be most translation-sensitive.
2. **Coverage tracking.** After each measured layout, it updates internal logs and computes uncovered or poorly sampled intervals in the selector domain. These gaps become the target for the next proposal.
3. **Adaptive next-step selection.** The next layout is chosen by focusing on the largest remaining gap, then adjusting the set of hugepages via page-add/remove moves that are informed by PEBS coverage and previous measurements.
4. **Fallback and repair logic.** If a candidate strategy cannot produce a useful novel layout, it falls back to alternate scan modes or local repair steps so the optimization keeps progressing rather than stalling.
5. **Stopping and resumption.** The loop continues until the layout budget is exhausted or the remaining gap becomes small enough; because state is persisted to CSV-backed logs, interrupted runs can resume without restarting the full search.

This makes Moselect especially appropriate when the objective is not just a single minimum but a model-quality objective: building a compact, informative set of measured points that supports accurate interpolation and prediction across the layout domain.

- Code path: `experiments/moselect/createLayouts.py`, `experiments/moselect/layout_generator.py`
- Typical run: `make moselect` or `make MOSELECT_NUM_LAYOUTS=50 MOSELECT_MAX_GAP=4 experiments/moselect`

## Bayesian Optimization Layout Selector
`experiments/bayesian_optimization/runExperiment.py` implements a Bayesian Optimization (BO) based search for the Mosalloc hugepage layout that minimizes a benchmark's second-level TLB (STLB) miss count. It is one of several layout-selection strategies provided by this repo (alongside, e.g., `moselect`, `genetic_selector`, and `pebs_selector`), and is invoked automatically by the `experiments/bayesian_optimization/module.mk` Makefile module.

### Problem formulation
A candidate memory layout is a decision of, for every 2MB-aligned region of the benchmark's `brk` heap, whether that region should be backed by a hugepage. This is naturally represented as a binary vector of length `num_hugepages` (one bit per region). The layout-selection problem is then: find the bit-vector that minimizes the benchmark's measured STLB misses, where each evaluation requires an actual (potentially expensive) benchmark run.

This is exactly the setting Bayesian Optimization is designed for: expensive, black-box, noisy objective functions where each evaluation should be used as efficiently as possible. We use [`scikit-optimize`](https://scikit-optimize.github.io/stable/)'s `gp_minimize`, which fits a Gaussian-process surrogate model over previously evaluated layouts and uses an acquisition function (Expected Improvement) to propose the next, most promising layout to try.

### Encoding hugepage layouts as a bounded search space
`gp_minimize` only supports a bounded number of numeric (`Integer`/`Real`) search dimensions, but a benchmark's `brk` heap can span thousands of 2MB regions -- far too many to encode as one dimension per region. `runExperiment.py` addresses this with a two-step encoding, following the general approach for Bayesian Optimization over binary vectors of [Baptista and Poloczek (2018)](https://arxiv.org/pdf/1807.02811.pdf):

1. **Gray-code transformation.** The hugepage bit-vector is converted to [reflected binary (Gray) code](https://en.wikipedia.org/wiki/Gray_code) before being handed to the optimizer. Gray code guarantees that incrementing or decrementing the encoded integer toggles exactly one bit, so layouts that are numerically close remain close in Hamming distance. This keeps the search space smooth enough for the Gaussian-process surrogate to interpolate meaningfully between sampled layouts.
2. **Dimension packing.** The (Gray-coded) bit-vector is split into fixed-size 64-bit chunks, each represented as a single `skopt.space.Integer` dimension, bounded by `BayesianExperiment.MAX_DIMENSIONS` (20 dimensions, i.e. up to 1279 hugepage regions). If a benchmark's footprint would require more regions than that, the *effective* hugepage size is transparently scaled up (grouping several 2MB hugepages into one logical region) until the encoding fits.

This compression scheme is implemented in `BayesianExperiment.compress_memory_layout` / `decompress_memory_layout` (see the in-code docstrings for the full bit-level walkthrough), and lets the same BO loop scale from small toy benchmarks to large-footprint SPEC/GAP benchmarks without changing the number of search dimensions exposed to `gp_minimize`.

### Initial sample ("warm start") strategies
Like most surrogate-model-based optimizers, BO benefits substantially from a well-chosen set of initial, pre-evaluated samples. `runExperiment.py` supports several strategies, selectable via `--initialization_method` (or the `BAYESIAN_INIT_METHOD` Makefile variable):

- `base` (default): only the two trivial extremes: no hugepages, and all hugepages.
- `random`: no explicit initial samples; `gp_minimize` chooses its own.
- `our_random`: a set of uniformly random hugepage layouts.
- `chebyshev`: layouts sampled at Chebyshev-node positions across the compressed integer search space, biasing samples toward the domain edges.
- `chebyshev_misses`: layouts targeting Chebyshev-node-spaced STLB-miss counts, derived from a static Intel PEBS-based TLB-miss estimate (requires PEBS sampling data).
- `moselect`: pages are grouped into three PEBS-TLB-coverage-weighted buckets (~56% / 28% / 14%), and all `2^3` subset combinations of the buckets are evaluated -- a static-profiling-guided warm start inspired by the MosSelect layout selector in this repo.

### Running it
This selector is normally driven end-to-end through `make`, e.g.:

```bash
make BAYESIAN_NUM_LAYOUTS=50 BAYESIAN_INIT_METHOD=moselect experiments/bayesian_optimization
```

which uses `experiments/bayesian_optimization/module.mk` to wire up the benchmark run/collect scripts, the memory footprint and PEBS inputs, and invoke `runExperiment.py` with the appropriate arguments. See the docstrings at the top of `experiments/bayesian_optimization/runExperiment.py` for the equivalent direct command-line invocation and a full description of each argument.

In practice, this selector is best when the goal is finding a low-miss layout quickly while still exploring enough of the space to avoid poor local choices.

- Code path: `experiments/bayesian_optimization/runExperiment.py`
- Typical run: `make bayesian` or `make BAYESIAN_NUM_LAYOUTS=50 BAYESIAN_INIT_METHOD=moselect experiments/bayesian_optimization`

## Genetic Selector
The genetic selector treats a hugepage layout as a chromosome: a binary string or equivalent bit-pattern representation of which heap regions are backed by hugepages. It repeatedly evolves a population of layouts, evaluates each candidate by running the benchmark and measuring its STLB misses or runtime, and keeps the strongest individuals for the next generation.

### How it works
1. **Population initialization.** A pool of candidate layouts is generated, usually by random sampling and sometimes with a small amount of structure or PEBS-guidance from the current workload.
2. **Fitness evaluation.** Every candidate is executed and scored according to the measured objective (e.g., STLB-miss count or a derived runtime metric), so each individual receives a numerical fitness.
3. **Selection and reproduction.** Higher-fitness layouts are more likely to reproduce; new candidates are created by combining parent layouts and then mutating them to explore nearby variants.
4. **Iterative improvement.** The process repeats over several generations, preserving promising layouts while injecting randomness to avoid converging too early to a local optimum.

This makes the genetic selector a useful model-free baseline: it does not assume a smooth surrogate model like BO, and it can search multiple promising regions of the layout space in parallel. It is especially relevant when the objective landscape is irregular or when a simple stochastic search is preferable to a more structured optimizer.

- Code path: `experiments/genetic_selector/`
- Typical run: `make genetic` or `make experiments/genetic_selector`

## PEBS Selector
The PEBS selector is a profiling-driven heuristic baseline. Instead of iteratively evaluating candidate layouts through the benchmark, it uses hardware-level memory-access sampling to estimate which heap regions are most important for translation performance, and then directly builds a hugepage layout from that ranking.

### How it works
1. **Profile memory-access density.** Intel PEBS is used to sample memory addresses and estimate which heap regions are accessed most often or are associated with the highest translation pressure.
2. **Rank/partition the heap.** The observed access patterns are converted into a per-region importance score; the most important regions are treated as prime candidates for hugepage backing.
3. **Construct a layout directly.** A static layout is then produced by prioritizing the high-score pages while keeping the rest in base-page mode, without waiting for iterative search feedback from multiple benchmark runs.
4. **Use as a fast baseline.** This approach is simple, deterministic, and very cheap compared to full optimization loops, but it is also approximate: it relies on static profiling and does not adapt after each measured experiment the way Moselect does.

The PEBS selector is therefore the repo's fastest and simplest heuristic baseline: it is excellent for deriving a strong, structurally informed layout quickly, but it does not exploit the same closed-loop refinement that coverage-based or surrogate-based methods use.

- Code path: `experiments/pebs_selector/`
- Typical run: `make experiments/pebs_selector`

# Limitations
- The flow is Linux-first and tuned/tested mainly on Ubuntu-like systems with Intel PMU/PEBS support; using other distributions or non-Intel CPUs may require script and tooling changes.
- The selectors focus on `brk`-pool hugepage layout effects; workloads dominated by other allocators or `mmap` behavior may need additional adaptation.
- Current layout modeling is centered on 4KiB and 2MiB page-size behavior; 1GiB-page-aware selection is not yet implemented end-to-end.
- Running full experiments can still be expensive because every selected layout requires real benchmark executions and repeated measurements.

