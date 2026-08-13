# Introduction
Mosmodel is primarily useful for architectural virtual memory studies. It allows researchers to construct mathematical models that predict the runtime of applications from their virtual memory performance (e.g., the L1/L2 TLB miss rate and the latency of page table walks). Such models are a key component in the partial simulation methodology, which architects use to predict the performance of a newly proposed virtual memory design. Mosmodel is built on top of Mosalloc, a new memory allocator for hugepages. This repo is fully automated and contains all required tools (including Mosalloc) to produce the Mosmodel for any workload on any x86-64 Linux system.

More details about Mosmodel and Mosalloc can be found in the [MICRO'20](https://www.microarch.org/micro53/) paper:
["Predicting execution times with partial simulations in virtual memory research: why and how"](https://www.cs.technion.ac.il/~dan/papers/mosalloc-micro-2020.pdf)
by Mohammad Agbarya, Idan Yaniv, Jayneel Gandhi, Dan Tsafrir 

# Quick Start
Simply clone this repo, enter the repo directory, and run `make`.
This will produce Mosmodel for the toy benchmark (random access over a 1GB array, takes ~15 seconds) provided in the repo.

# Software Prerequisites
- **Sudo permissions**: multiple steps require sudo priviliges, most notably, reserving hugepages and installing apt packages (perf, numactl, ...). We recommend to [configure sudo permissions without password](https://www.cyberciti.biz/faq/linux-unix-running-sudo-command-without-a-password/) to fully automate the workflow. Sudo with password may stop the workflow at these steps prompting for your password.
- **Linux distro**: The code was tested on Ubuntu 20 LTS. Please note that all necessary apt packages are downloaded automatically. If you are using a different Linux distribution, you should probably modify the makefile to use the proper package management software and package names.
- **Python**: Our scripts are written in Python3 and rely on python packages like numpy, pandas, matplotlib, scipy, scikit-optimize, bitarray, and scikit-learn (see `requirements.txt`). Run `make install-prereqs` to automatically install the required system packages (`python3`, `python3-venv`, `python3-pip`) and set up an isolated Python virtual environment at `.venv` with every package from `requirements.txt` (see `scripts/install_prereqs.sh`); then activate it with `source .venv/bin/activate` before running any experiments. Re-running `make install-prereqs` is safe/idempotent -- it reuses the existing `.venv` if one is already present. Alternatively, you may use your own Python3 environment (e.g., [Anaconda](https://www.anaconda.com/products/individual)) as long as it has the packages listed in `requirements.txt` installed.

# Hardware Prerequisites
- **Intel CPUs**: Mosmodel collects and analyzes the hardware performance counters of Intel CPUs. Additionally, our code uses Intel Precise Event Based Sampling (PEBS) to find interesting Mosalloc layouts. We successfully tested our code on Intel Broadwell processors or newer.

# Setup and Configuration
Before you start building and running Mosmodel, you need to set and configure the following:
- Update the following variables in `benchmark.mk`:
    - `BENCHMARK_PATH`: the full path to a directory containing the benchmark. This directory must contain the following files:
        - `pre_run.sh` - a script running before the "actual" benchmark is measured.
        - `run.sh` - the main script of the benchmark which will be measured with perf.
        - `post_run.sh` - a script running after the "actual" benchmark is measured.

# Mosmodel Directory Structure
- `mosalloc` - a git submodule pointing to the Mosalloc memory allocator.
- `scripts` - python scripts to run the experiments, collect the results, build Mosmodel and everything in between.
- `experiments` - every experiment (== a single run of the benchmark) is stored under this directory.
- `analysis` - CSV files with raw data and the model coefficients.
- `toy_benchmark` - a small-memory benchmark supplied with this repo to quickly demonstrate Mosmodel and how it is built. It allocates a 1GB array and reads it randomly.
- `client_server_example` - a demo of how to create a benchmark infrastructure (`pre_run.sh`, `run.sh`, `post_run.sh`) for a client-server workloads, e.g., memcached.

# Bayesian Optimization Layout Selector
`experiments/bayesian_optimization/runExperiment.py` implements a Bayesian Optimization (BO) based search for the Mosalloc hugepage layout that minimizes a benchmark's second-level TLB (STLB) miss count. It is one of several layout-selection strategies provided by this repo (alongside, e.g., `moselect`, `genetic_selector`, `pebs_selector`), and is invoked automatically by the `experiments/bayesian_optimization/module.mk` Makefile module.

## Problem formulation
A candidate memory layout is a decision of, for every 2MB-aligned region of the benchmark's `brk` heap, whether that region should be backed by a hugepage. This is naturally represented as a binary vector of length `num_hugepages` (one bit per region). The layout-selection problem is then: find the bit-vector that minimizes the benchmark's measured STLB misses, where each evaluation requires an actual (potentially expensive) benchmark run.

This is exactly the setting Bayesian Optimization is designed for: expensive, black-box, noisy objective functions where each evaluation should be used as efficiently as possible. We use [`scikit-optimize`](https://scikit-optimize.github.io/stable/)'s `gp_minimize`, which fits a Gaussian-process surrogate model over previously evaluated layouts and uses an acquisition function (Expected Improvement) to propose the next, most promising layout to try.

## Encoding hugepage layouts as a bounded search space
`gp_minimize` only supports a bounded number of numeric (`Integer`/`Real`) search dimensions, but a benchmark's `brk` heap can span thousands of 2MB regions -- far too many to encode as one dimension per region. `runExperiment.py` addresses this with a two-step encoding, following the general approach for Bayesian Optimization over binary vectors of [Baptista and Poloczek (2018)](https://arxiv.org/pdf/1807.02811.pdf):

1. **Gray-code transformation.** The hugepage bit-vector is converted to [reflected binary (Gray) code](https://en.wikipedia.org/wiki/Gray_code) before being handed to the optimizer. Gray code guarantees that incrementing/decrementing the encoded integer toggles exactly one bit, so layouts that are numerically close remain close in Hamming distance. This keeps the search space smooth enough for the Gaussian-process surrogate to interpolate meaningfully between sampled layouts.
2. **Dimension packing.** The (Gray-coded) bit-vector is split into fixed-size 64-bit chunks, each represented as a single `skopt.space.Integer` dimension, bounded by `BayesianExperiment.MAX_DIMENSIONS` (20 dimensions, i.e. up to 1279 hugepage regions). If a benchmark's footprint would require more regions than that, the *effective* hugepage size is transparently scaled up (grouping several 2MB hugepages into one logical region) until the encoding fits.

This compression scheme is implemented in `BayesianExperiment.compress_memory_layout` / `decompress_memory_layout` (see the in-code docstrings for the full bit-level walkthrough), and lets the same BO loop scale from small toy benchmarks to large-footprint SPEC/GAP benchmarks without changing the number of search dimensions exposed to `gp_minimize`.

## Initial sample ("warm start") strategies
Like most surrogate-model-based optimizers, BO benefits substantially from a well-chosen set of initial, pre-evaluated samples. `runExperiment.py` supports several strategies, selectable via `--initialization_method` (or the `BAYESIAN_INIT_METHOD` Makefile variable):

| Strategy | Description |
|---|---|
| `base` (default) | Only the two trivial extremes: no hugepages, and all hugepages. |
| `random` | No explicit initial samples; `gp_minimize` chooses its own. |
| `our_random` | A set of uniformly random hugepage layouts. |
| `chebyshev` | Layouts sampled at Chebyshev-node positions across the compressed integer search space, biasing samples toward the domain edges. |
| `chebyshev_misses` | Layouts targeting Chebyshev-node-spaced STLB-miss counts, derived from a static Intel PEBS-based TLB-miss estimate (requires PEBS sampling data). |
| `moselect` | Pages are grouped into three PEBS-TLB-coverage-weighted buckets (~56%/28%/14%), and all `2^3` subset combinations of the buckets are evaluated -- a static-profiling-guided warm start inspired by the MosSelect layout selector in this repo. |

## Running it
This selector is normally driven end-to-end through `make`, e.g.:
```
make BAYESIAN_NUM_LAYOUTS=50 BAYESIAN_INIT_METHOD=moselect experiments/bayesian_optimization
```
which uses `experiments/bayesian_optimization/module.mk` to wire up the benchmark run/collect scripts, the memory footprint and PEBS inputs, and invoke `runExperiment.py` with the appropriate arguments. See the docstrings at the top of `experiments/bayesian_optimization/runExperiment.py` for the equivalent direct command-line invocation and a full description of each argument.

## Software Prerequisites
In addition to the general [Software Prerequisites](#software-prerequisites) above, this selector requires the `scikit-optimize` (`skopt`) and `bitarray` Python packages; both are listed in `requirements.txt` and installed automatically by `make install-prereqs`.

# MosSelect Layout Selector
`experiments/moselect/layout_generator.py` (driven by `experiments/moselect/createLayouts.py`) implements MosSelect, an adaptive, PEBS-guided hugepage layout selector. Unlike the [Bayesian Optimization selector](#bayesian-optimization-layout-selector), which searches for the single *best* layout, MosSelect's goal is to pick a fixed-size *set* of layouts whose measured runtimes are evenly spread across the full performance range -- from the slowest (no hugepages) to the fastest (all hugepages) -- so that the resulting scatter plot (e.g. `make analysis/moselect/normalized_scatter.pdf`) and any regression fitted to it (Mosmodel) are well-conditioned. It is invoked once per layout by `experiments/moselect/module.mk`, resuming from CSV-backed state on each invocation so the search can proceed incrementally as `make` measures each newly generated layout.

## Problem formulation
As with the other selectors, a candidate layout is a choice of which 2MB-aligned regions of the benchmark's `brk` heap to back with hugepages. MosSelect measures a layout's position along the performance range using two coverage metrics:

| Term | Meaning |
|---|---|
| **PEBS coverage** | The *statically predicted* percentage (0-100) of sampled TLB accesses a candidate hugepage set is expected to serve, computed once from an Intel PEBS sampling pass over the benchmark (no need to run the benchmark itself). |
| **Real coverage** | The *actually measured* position of a layout, expressed as 0-100% between the slowest measured layout (highest STLB `walk_cycles`) and the fastest (lowest `walk_cycles`). Only known after a layout has been executed. |
| **Gap** | The real-coverage difference between two adjacent (by real coverage) measured layouts. MosSelect's target is to keep every such gap below `--max_gap` (`MOSELECT_MAX_GAP`, default 4 percentage points). |

Given a fixed total budget of layouts to measure (`--max_budget` / `MOSELECT_NUM_LAYOUTS`, default 50), MosSelect must decide, at each step, which not-yet-tried hugepage set is most likely to close the largest remaining real-coverage gap -- using only the cheap PEBS coverage as a proxy, since the real coverage of a candidate is unknown until it is measured.

## Algorithm: static bootstrap + adaptive gap closing
MosSelect operates in two phases, implemented across `LayoutGenerator` (see the module docstring in `layout_generator.py` for the full algorithmic walkthrough) and persisted via `SubgroupsLog`/`StateLog` (`logs.py`):

1. **Static bootstrap** (`layout1`-`layout9`): PEBS-sampled pages are greedily partitioned into three coverage-weighted buckets (~56%/28%/14% of total PEBS coverage). All `2^3 = 8` subset combinations of the three buckets are evaluated as layouts, plus one additional all-hugepages layout -- 9 "anchor" layouts total, expected to roughly span the full performance range.
2. **Adaptive gap closing** (`layout10` onward): once the anchors are measured, each of the 8 intervals between adjacent anchors ("subgroups") is allocated a budget of additional layouts, proportional to the size of its real-coverage gap (`SubgroupsLog.calculateBudget`). While budget remains, MosSelect repeatedly targets the subgroup's largest remaining gap and proposes a new candidate layout to bisect it, trying strategies in order of preference: adding PEBS-coverage-targeted pages (`add`), removing pages (`remove`), adding head-order pages (`add_round2`), and, only if all data-driven strategies fail to produce a novel layout, blind heuristic page mixing (`auto_reduce-max` / `auto_blind`). Once every subgroup's gaps are within `max_gap`, any leftover budget is spent closing the single largest remaining gap across all subgroups.

Because each invocation only produces one layout and then exits, all of this state (per-subgroup budgets, per-layout scan history, and page lists) is persisted to CSV files under the experiment directory (`subgroups.log`, `<right>_<left>_state.log`, `layout_pages.log`), making the search fully resumable across independent `make`-triggered invocations.

## Running it
This selector is normally driven end-to-end through `make`, e.g.:
```
make MOSELECT_NUM_LAYOUTS=50 MOSELECT_MAX_GAP=4 experiments/moselect
```
which uses `experiments/moselect/module.mk` to collect intermediate results after each layout, and invoke `createLayouts.py` with the memory footprint, PEBS mem-bins, and results file for the next layout to generate. See the docstrings in `experiments/moselect/createLayouts.py` for the equivalent direct command-line invocation and a full description of each argument.

## Software Prerequisites
This selector only relies on `pandas` (already listed in the general [Software Prerequisites](#software-prerequisites) / `requirements.txt`) and the Python standard library; no additional packages are required.

# Limitations (Future Work)
- Currently, Mosmodel scans only Mosalloc layouts on the `brk()` pool because it assumes that the benchmark allocates memory through `malloc()`. In case the benchmark uses different allocators (than glibc `malloc()`), then this assumption may not hold. We need to customize the python scripts and makefile infrastructure that create the Mosalloc layouts. The first step toward this goal is measuring the relative performance impact of hugepages in the `mmap()` and `brk()` pools, respectively.

