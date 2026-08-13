#!/usr/bin/env python3
"""Bayesian-optimization-based hugepage layout selector.

This script drives an end-to-end Bayesian Optimization (BO) loop that searches
for a memory (hugepage) layout minimizing the number of STLB (second-level
TLB) misses of a benchmark, following the general BO-over-binary-vectors
approach described in https://arxiv.org/pdf/1807.02811.pdf.

Problem encoding
----------------
A candidate memory layout is represented as a bit-vector of length
``num_hugepages``, where bit ``i`` indicates whether the ``i``-th 2MB-aligned
region of the benchmark's ``brk`` heap should be backed by a hugepage. This
vector can be exponentially long (thousands of bits for large-footprint
benchmarks), which is unsuitable for `scikit-optimize`'s `gp_minimize`, since
it only supports a bounded number of ``Integer``/``Real`` dimensions.

To work around this, the bit-vector is:
  1. Converted to reflected binary (Gray) code, so that layouts which differ
     by a single hugepage toggle remain close in Hamming distance -- this
     keeps the search space "smooth" for the Gaussian-process surrogate model
     used internally by `gp_minimize`.
  2. Packed into a small, fixed number of 64-bit integer "dimensions"
     (bounded by ``MAX_DIMENSIONS``), each treated as a single
     `skopt.space.Integer` search dimension.

This compression/decompression logic (see `compress_memory_layout` and
`decompress_memory_layout`) is what allows BO to scale to benchmarks with a
large `brk` footprint while keeping the dimensionality of the search space
manageable.

Initial samples
----------------
BO's efficiency benefits from a good "warm start". This module supports
several strategies for producing the initial set of evaluated layouts
(selected via ``--initialization_method``):
  - ``base``: only the no-hugepages and all-hugepages layouts.
  - ``random``: no explicit initial samples; `gp_minimize` picks them.
  - ``our_random``: uniformly random hugepage bit-vectors.
  - ``chebyshev``: Chebyshev-node-spaced samples over the compressed integer
    search space, giving denser sampling near the domain edges.
  - ``chebyshev_misses``: Chebyshev-node-spaced samples over the *expected*
    PEBS-predicted TLB-miss range (requires ``--pebs_mem_bins``).
  - ``moselect``: groups pages into weighted "buckets" by PEBS TLB coverage
    and evaluates all subsets of the resulting groups (see the MosSelect
    paper's static profiling approach).

Usage
-----
This script is normally invoked by
`experiments/bayesian_optimization/module.mk` as part of the `make`-driven
experiment pipeline; see that module and the project README for the full
Makefile target (`analysis/bayesian_optimization/...`) and prerequisites.
It can also be run directly, e.g.:

    ./runExperiment.py \\
        --memory_footprint=memory_footprint.csv \\
        --pebs_mem_bins=mem_bins_2mb.csv \\
        --exp_root_dir=experiments/bayesian_optimization \\
        --results_file=results/bayesian_optimization/median.csv \\
        --collect_reults_cmd=./collect_results.sh \\
        --run_experiment_cmd=./run_benchmark.sh \\
        --num_layouts=50 \\
        --initialization_method=base
"""
# import cProfile  # kept for local profiling; enable manually if needed (see __main__)
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Space
from skopt.utils import use_named_args
import itertools
from numpy.polynomial.chebyshev import chebgauss
import numpy as np
from bitarray import bitarray
import subprocess
import math
import os, sys

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
experiments_root_dir = os.path.join(curr_file_dir, '..')
sys.path.append(experiments_root_dir)
from Utils.utils import Utils

class BayesianExperiment:
    """Drives a single Bayesian Optimization experiment for one benchmark.

    An instance encapsulates: (1) the compressed Gray-code search space over
    hugepage layouts (see `prepare_space`), (2) the initial-sample generation
    strategies, and (3) the `run` loop that repeatedly evaluates candidate
    layouts on the real benchmark (via `run_experiment_cmd`) and feeds the
    measured STLB-miss count back into `skopt.gp_minimize` as the objective
    value to minimize.

    Based on the binary-vector Bayesian Optimization approach described in
    https://arxiv.org/pdf/1807.02811.pdf.

    Class attributes:
        MAX_DIMENSIONS: Upper bound on the number of `skopt.space.Integer`
            dimensions used to encode a layout. Each dimension packs up to
            `dimension_size_in_bits` (64) bits of the Gray-coded hugepage
            bit-vector, so the largest representable bit-vector length is
            ``MAX_DIMENSIONS * 64 - 1`` bits.
        DEFAULT_HUGEPAGE_SIZE: The baseline hugepage size in bytes (2MB).
            When a benchmark's footprint would require more hugepages than
            `MAX_DIMENSIONS` can encode, `prepare_space` transparently
            increases the effective hugepage size (i.e., groups multiple
            2MB hugepages together) so the compressed representation still
            fits.
    """
    MAX_DIMENSIONS = 20
    DEFAULT_HUGEPAGE_SIZE = 1 << 21 # 2MB

    def __init__(self,
                 memory_footprint_file, pebs_mem_bins_file,
                 collect_reults_cmd, results_file,
                 run_experiment_cmd, exp_root_dir,
                 num_layouts) -> None:
        """Initializes the experiment and builds the compressed search space.

        Args:
            memory_footprint_file: CSV path with columns ``anon-mmap-max``
                and ``brk-max``, describing the benchmark's peak memory
                footprint (as produced by `scripts/collectMemoryFootprint.py`).
            pebs_mem_bins_file: Optional CSV path with per-page PEBS TLB-miss
                sample counts (columns ``PAGE_NUMBER``/``NUM_ACCESSES``),
                used by the ``chebyshev_misses``/``moselect`` initialization
                strategies and by `predictTlbMisses`. May be ``None`` if
                those strategies/estimates are not needed.
            collect_reults_cmd: Shell command that aggregates the raw
                per-layout benchmark measurements into `results_file`.
            results_file: CSV path where aggregated per-layout results
                (including the ``stlb_misses``/``cpu_cycles`` columns) are
                written by `collect_reults_cmd`.
            run_experiment_cmd: Shell command template that runs the
                benchmark for a single layout (the layout name is appended
                as its last argument; see `run_workload`).
            exp_root_dir: Root directory of this experiment, under which
                per-layout output directories and the ``layouts/`` folder
                (holding the generated Mosalloc configuration CSVs) live.
            num_layouts: Total number of layouts (initial + BO-selected) to
                evaluate before stopping.
        """
        self.last_layout_num = 0
        self.collect_reults_cmd = collect_reults_cmd
        self.results_file = results_file
        self.memory_footprint_file = memory_footprint_file
        self.pebs_mem_bins_file = pebs_mem_bins_file
        self.run_experiment_cmd = run_experiment_cmd
        self.exp_root_dir = exp_root_dir
        self.num_layouts = num_layouts
        self.prepare_space()

    def prepare_space(self):
        """Builds the compressed Gray-code search space for `gp_minimize`.

        This method determines, from the benchmark's memory footprint, how
        many 2MB hugepages span the `brk` heap (`num_hugepages`) and encodes
        the corresponding hugepage bit-vector (plus one extra bit needed by
        the binary-to-Gray-code conversion) into the smallest number of
        64-bit `skopt.space.Integer` dimensions that fit within
        `MAX_DIMENSIONS`.

        If the natural per-hugepage encoding would require more than
        `MAX_DIMENSIONS` dimensions, the effective hugepage size is scaled up
        (grouping consecutive 2MB hugepages into one "compressed hugepage",
        see `hugepages_in_compressed_hugepage`) until the bit-vector fits.
        The benchmark's footprint is then rounded up to a multiple of the
        (possibly enlarged) hugepage size so that layout boundaries stay
        aligned.

        Populates (among others) the following instance attributes used
        throughout the class: `num_hugepages`, `hugepage_size`,
        `hugepages_in_compressed_hugepage`, `num_dimensions`, `dimensions`
        (the list of `skopt.space.Integer` objects passed to `gp_minimize`),
        and, if `pebs_mem_bins_file` was provided, `pebs_df`/`total_misses`.
        """
        # read memory-footprints
        self.footprint_df = pd.read_csv(self.memory_footprint_file)
        self.mmap_footprint = self.footprint_df['anon-mmap-max'][0]
        self.brk_footprint = self.footprint_df['brk-max'][0]
        self.memory_footprint = self.brk_footprint

        self.hugepage_size = BayesianExperiment.DEFAULT_HUGEPAGE_SIZE
        self.num_hugepages = math.ceil(self.memory_footprint / self.hugepage_size) # bit vector length
        self.num_default_hugepages = math.ceil(self.memory_footprint / BayesianExperiment.DEFAULT_HUGEPAGE_SIZE)

        # scikit-optimize and NumPy integer paths are signed-int based.
        # Keep each dimension within int64 range to avoid overflow in ask()/rvs().
        self.dimension_size_in_bits = 63
        self.dimension_capacity = 2**self.dimension_size_in_bits
        # the num_dimensions is calculated for (num_hugepages + 1) because
        # an additional bit may be required when converting a binary number to gray code
        self.num_dimensions = math.ceil((self.num_hugepages + 1) / self.dimension_size_in_bits)
        if self.num_dimensions > BayesianExperiment.MAX_DIMENSIONS:
            self.num_dimensions = BayesianExperiment.MAX_DIMENSIONS
            # length(gray_code) = length(bit_vector) - 1
            self.max_num_hugepages = (BayesianExperiment.MAX_DIMENSIONS * self.dimension_size_in_bits) - 1
            self.hugepage_size = Utils.round_up(
                math.ceil(self.memory_footprint / self.max_num_hugepages),
                BayesianExperiment.DEFAULT_HUGEPAGE_SIZE)
            self.num_hugepages = math.ceil(self.memory_footprint / self.hugepage_size)
        # update num_dimensions and layout_bit_vector_length in case we exceeded the MAX_DIMESNIONS
        self.layout_bit_vector_length = self.num_hugepages
        self.gray_layout_bit_vector_length = self.layout_bit_vector_length + 1
        self.num_dimensions = math.ceil(self.gray_layout_bit_vector_length / self.dimension_size_in_bits)
        self.hugepages_in_compressed_hugepage = self.hugepage_size // BayesianExperiment.DEFAULT_HUGEPAGE_SIZE
        # Define the search space
        self.dimension_min_val = 0
        self.dimension_max_val = self.dimension_capacity - 1
        self.last_dimension_size_in_bits = self.gray_layout_bit_vector_length % self.dimension_size_in_bits
        if self.last_dimension_size_in_bits == 0:
            self.last_dimension_size_in_bits = self.dimension_size_in_bits
        self.last_dimension_max_val = (2**self.last_dimension_size_in_bits) - 1
        self.dimensions = [Integer(self.dimension_min_val, self.dimension_max_val, name=f'mem_region_{i}') for i in range(self.num_dimensions - 1)]
        self.dimensions += [Integer(self.dimension_min_val, self.last_dimension_max_val, name=f'mem_region_{self.num_dimensions-1}')]

        # round up the memory footprint to match the new boundaries of the new hugepage-size
        self.memory_footprint = (self.num_hugepages + 1) * self.hugepage_size
        self.brk_footprint = self.memory_footprint

        if self.pebs_mem_bins_file is None:
            print('pebs_mem_bins_file argument is missing, skipping loading PEBS results...')
            self.pebs_df = None
            self.total_misses = None
        else:
            self.pebs_df = Utils.load_pebs(self.pebs_mem_bins_file, True)
            self.total_misses = self.pebs_df['NUM_ACCESSES'].sum()

        if False:
            print(f'self.layout_bit_vector_length={self.layout_bit_vector_length}')
            print(f'self.gray_layout_bit_vector_length={self.gray_layout_bit_vector_length}')
            print(f'self.num_dimensions={self.num_dimensions}')
            print(f'self.hugepages_in_compressed_hugepage={self.hugepages_in_compressed_hugepage}')
            print(f'self.last_dimension_size_in_bits={self.last_dimension_size_in_bits}')
            print(f'self.num_hugepages={self.num_hugepages}')
            print(f'self.hugepage_size={self.hugepage_size}')
            print(f'self.memory_footprint={self.memory_footprint}')

    def run_command(command, out_dir):
        """Runs a shell command, logging its stdout/stderr under `out_dir`.

        Args:
            command: The shell command line to execute (run with
                ``shell=True``).
            out_dir: Directory to write ``benchmark.log`` into; created if
                missing.

        Returns:
            int: The subprocess's return code (0 on success). On failure,
            the command, its output and error streams are printed to help
            diagnose the failed experiment run.
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Run the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        # Get the output and error messages
        output = output.decode('utf-8')
        error = error.decode('utf-8')

        # Check the return code
        return_code = process.returncode

        output_log = f'{out_dir}/benchmark.log'
        error_log = f'{out_dir}/benchmark.log'
        with open(output_log, 'w+') as out:
            out.write(output)
            out.write('============================================')
            out.write(f'the process exited with status: {return_code}')
            out.write('============================================')
        with open(error_log, 'w+') as err:
            err.write(error)
            err.write('============================================')
            err.write(f'the process exited with status: {return_code}')
            err.write('============================================')
        if return_code != 0:
            # Print the output and error
            print('============================================')
            print(f'Failed to run the following command with exit code: {return_code}')
            print(f'Command line: {command}')
            print('Output:', output)
            print('Error:', error)
            print('Return code:', return_code)
            print('============================================')

        return return_code

    def collect_results(collect_reults_cmd, results_file):
        """Runs `collect_reults_cmd` and loads the resulting results table.

        Args:
            collect_reults_cmd: Shell command that (re)generates
                `results_file` from the raw per-layout measurements.
            results_file: Path to the aggregated results CSV produced by
                `collect_reults_cmd`.

        Returns:
            pandas.DataFrame: The aggregated results (one row per evaluated
            layout), or an empty DataFrame if `results_file` does not exist
            yet (e.g., on the very first call, before any layout has run).

        Raises:
            RuntimeError: If `collect_reults_cmd` exits with a non-zero
                status.
        """
        print(f'** collecting results: {collect_reults_cmd}')

        # Extract the directory path
        results_dir = os.path.dirname(results_file)
        # Create the directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        ret_code = BayesianExperiment.run_command(collect_reults_cmd, results_dir)
        if ret_code != 0:
            raise RuntimeError(f'Error: collecting experiment results failed with error code: {ret_code}')
        if os.path.exists(results_file):
            results_df = Utils.load_dataframe(results_file)
        else:
            results_df = pd.DataFrame()

        return results_df

    def convert_to_gray(binary):
        """Converts a binary bit-vector to reflected binary (Gray) code.

        Gray code guarantees that incrementing/decrementing the underlying
        integer by one flips exactly one bit, which keeps "nearby" layouts
        close together in Hamming distance -- a property the Gaussian-process
        surrogate model inside `gp_minimize` relies on when interpolating
        between sampled layouts.

        Args:
            binary: A `bitarray`, a binary string (e.g. ``'1011'``), or a
                non-negative integer to be interpreted as a binary number.

        Returns:
            bitarray: The Gray-code encoding of `binary`, same length as the
            input bit-vector.
        """
        if isinstance(binary, str):
            binary = bitarray(binary)
        elif str(binary).isnumeric():
            binary = bitarray(bin(binary)[2:])
        gray = bitarray(0)
        gray.append(binary[0])
        for i in range(1, len(binary)):
            gray.append(binary[i] ^ binary[i-1])
        return gray

    def set_bits(bitarray_obj, bits_val):
        """Returns the bitwise-OR of `bitarray_obj` with the bits of `bits_val`.

        Both operands are zero-padded (on the left) to the same length
        before the OR is applied. Currently unused by the main pipeline
        (superseded by direct index assignment in
        `convert_mem_layout_to_gray`); kept as a documented utility.

        Args:
            bitarray_obj: The base `bitarray` (or bit-string) to update.
            bits_val: An integer whose binary representation's set bits
                should be OR-ed into `bitarray_obj`.

        Returns:
            bitarray: The resulting bit-vector after the OR operation.
        """
        bits_to_set = bin(bits_val)[2:]
        bitarray_bits = bitarray_obj.to01()
        max_len = max(len(bits_to_set), len(bitarray_bits))
        bits_to_set = bits_to_set.zfill(max_len)
        bitarray_bits = bitarray_bits.zfill(max_len)
        new_bitarray = bitarray(bits_to_set) | bitarray(bitarray_bits)
        return new_bitarray

    def convert_from_gray(gray):
        """Converts a Gray-coded bit-vector back to standard binary.

        This is the inverse of `convert_to_gray`: the most-significant bit is
        copied as-is, and each subsequent binary bit is the XOR of the
        previous binary bit with the corresponding Gray-code bit.

        Args:
            gray: A `bitarray`, binary string, or integer representing the
                Gray-coded value.

        Returns:
            bitarray: The decoded binary bit-vector, same length as `gray`.
        """
        if isinstance(gray, str):
            gray = bitarray(gray)
        elif str(gray).isnumeric():
            gray = bitarray(bin(gray)[2:])
        binary = bitarray(0)
        binary.append(gray[0])
        for i in range(1, len(gray)):
            binary.append(binary[i-1] ^ gray[i])
        return binary

    def convert_mem_layout_to_gray(self, mem_layout_hugepages):
        """Encodes a list of hugepage indices as a Gray-coded bit-vector.

        Args:
            mem_layout_hugepages: Iterable of (base, i.e. 2MB-granularity)
                hugepage indices that should be backed by a hugepage in this
                layout.

        Returns:
            bitarray: The Gray-coded bit-vector of length
            `gray_layout_bit_vector_length`, ready to be sliced into
            per-dimension integers by `compress_memory_layout`.
        """
        mem_layout_bin = bitarray(self.gray_layout_bit_vector_length)
        mem_layout_bin.setall(0)
        # createa one long bit-vector that represents the memory layout
        for p in mem_layout_hugepages:
            # mem_layout_bin = set_bits(mem_layout_bin, p)
            aggregated_p = int(p // self.hugepages_in_compressed_hugepage)
            mem_layout_bin[aggregated_p] = 1
        # reverse the string to make it readable as binary string
        mem_layout_bin.reverse()
        # convert to gray-code
        gray_mem_layout = BayesianExperiment.convert_to_gray(mem_layout_bin)
        gray_mem_layout.reverse()
        return gray_mem_layout

    def convert_dimensions_to_mem_layout_bin(self, mem_layout_dimensions):
        """Inverse of `compress_memory_layout`'s packing step.

        Reassembles the per-dimension integers produced by `gp_minimize`
        (or by the initial-sample generators) back into a single Gray-coded
        bit-vector, then decodes it to standard binary.

        Args:
            mem_layout_dimensions: Sequence of `num_dimensions` integers, one
                per `skopt.space.Integer` search dimension.

        Returns:
            bitarray: The decoded (non-Gray) hugepage bit-vector.
        """
        gray_mem_layout = bitarray(0)
        for i in range(len(mem_layout_dimensions)):
            gray_word = bin(mem_layout_dimensions[i])[2:]
            padding_size = self.dimension_size_in_bits
            if i == (len(mem_layout_dimensions) - 1):
                padding_size = self.last_dimension_size_in_bits
            padded_word = gray_word.zfill(padding_size)
            gray_mem_layout.extend(padded_word)
        gray_mem_layout.reverse()
        mem_layout = BayesianExperiment.convert_from_gray(gray_mem_layout)
        mem_layout.reverse()
        return mem_layout

    def decompress_memory_layout(self, mem_layout_dimensions):
        """Converts a compressed (per-dimension integer) layout to hugepage indices.

        Args:
            mem_layout_dimensions: Sequence of `num_dimensions` integers as
                produced/consumed by `gp_minimize` (the point ``x`` in the
                search space).

        Returns:
            list[int]: The base (2MB-granularity) hugepage indices that are
            set in this layout, expanded from any "compressed hugepage"
            grouping (`hugepages_in_compressed_hugepage`) back to individual
            2MB hugepages. Suitable for passing to `Utils.write_layout`.
        """
        hugepages_bit_vector = self.convert_dimensions_to_mem_layout_bin(mem_layout_dimensions)
        mem_layout_hugepages = []
        for i in range(len(hugepages_bit_vector)):
            if hugepages_bit_vector[i] == 1:
                for k in range(self.hugepages_in_compressed_hugepage):
                    hugepage_idx = i * self.hugepages_in_compressed_hugepage + k
                    mem_layout_hugepages.append(hugepage_idx)
        return mem_layout_hugepages

    def compress_memory_layout(self, mem_layout_hugepages):
        """Encodes a list of hugepage indices into `gp_minimize`'s point format.

        This is the forward path used both to seed initial samples (``x0``)
        and to record newly evaluated layouts: the hugepage bit-vector is
        Gray-coded (`convert_mem_layout_to_gray`) and then sliced into
        `num_dimensions` fixed-width integers, one per search dimension.

        Args:
            mem_layout_hugepages: Iterable of base hugepage indices set in
                this layout.

        Returns:
            list[int]: A point ``x`` of length `num_dimensions`, suitable as
            an element of `gp_minimize`'s ``x0`` list or as the argument to
            `objective_function`.
        """
        gray_mem_layout = self.convert_mem_layout_to_gray(mem_layout_hugepages)

        compressed_mem_layout = [0] * self.num_dimensions
        for i in range(self.num_dimensions):
            dimension_start_idx = i*self.dimension_size_in_bits
            dimension_end_idx = dimension_start_idx + self.dimension_size_in_bits
            if dimension_start_idx >= len(gray_mem_layout):
                print('WARNING: memory layout size in gray code is smaller than in normal binary code')
                sys.exit(1)
                break
            if i == (self.num_dimensions - 1):
                dimension_end_idx = dimension_start_idx + self.last_dimension_size_in_bits
            gray_i = gray_mem_layout[dimension_start_idx:dimension_end_idx]
            gray_i.reverse()
            gray_i_number = int(gray_i.to01(), 2)
            compressed_mem_layout[i] = gray_i_number

        return compressed_mem_layout

    def predictTlbMisses(self, mem_layout):
        """Estimates STLB misses of `mem_layout` using static PEBS profiling.

        This is a *static* estimate (no benchmark execution): it assumes that
        every page in `mem_layout` that is backed by a hugepage eliminates
        its PEBS-sampled share of TLB misses, proportionally to that page's
        `NUM_ACCESSES` share of the total. It is used as a fast surrogate to
        find PEBS-coverage thresholds for the ``chebyshev_misses``
        initialization strategy, not as the objective function itself (which
        always measures misses on real hardware, see `run_workload`).

        Args:
            mem_layout: Iterable of PEBS page numbers assumed to be backed by
                hugepages.

        Returns:
            float: The estimated remaining STLB misses (`total_misses` minus
            the PEBS coverage of `mem_layout`).

        Raises:
            AssertionError: If `pebs_df` was not loaded (i.e.,
                `pebs_mem_bins_file` was ``None``).
        """
        assert self.pebs_df is not None
        expected_tlb_coverage = self.pebs_df.query(f'PAGE_NUMBER in {mem_layout}')['NUM_ACCESSES'].sum()
        expected_tlb_misses = self.total_misses - expected_tlb_coverage
        print(f'[DEBUG]: mem_layout of size {len(mem_layout)} has an expected-tlb-coverage={expected_tlb_coverage} and expected-tlb-misses={expected_tlb_misses}')
        return expected_tlb_misses

    def generate_layout_from_pebs(self, pebs_coverage):
        """Greedily builds a layout targeting a given PEBS TLB-coverage percentage.

        Pages are added in decreasing order of `TLB_COVERAGE` until the
        accumulated coverage reaches `pebs_coverage` (within 0.5 percentage
        points), giving the smallest layout expected to cover roughly
        `pebs_coverage`% of the sampled TLB accesses.

        Args:
            pebs_coverage: Target cumulative TLB-coverage percentage
                (0-100) to reach.

        Returns:
            list[int]: PEBS page numbers to back with hugepages.

        Raises:
            AssertionError: If `pebs_df` was not loaded.
        """
        assert self.pebs_df is not None

        df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)

        mem_layout = []
        total_weight = 0
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            if (total_weight + weight) < (pebs_coverage + 0.5):
                mem_layout.append(page)
                total_weight += weight
            if total_weight >= pebs_coverage:
                break
        return mem_layout

    def get_layout_results(self, layout_name):
        """Looks up the measured STLB misses and runtime for `layout_name`.

        Re-collects the results table (via `collect_results`) and extracts
        the row matching `layout_name`.

        Args:
            layout_name: Name of the previously executed layout (e.g.
                ``'layout3'``).

        Returns:
            tuple[float, float]: ``(tlb_misses, runtime)`` -- the
            ``stlb_misses`` and ``cpu_cycles`` columns for this layout's row.
        """
        results_df = BayesianExperiment.collect_results(self.collect_reults_cmd, self.results_file)
        layout_results = results_df[results_df['layout'] == layout_name]
        tlb_misses = layout_results['stlb_misses'].iloc[0]
        runtime = layout_results['cpu_cycles'].iloc[0]
        return tlb_misses, runtime

    def fill_buckets(self, buckets_weights, start_from_tail=False, fill_min_buckets_first=True):
        """Greedily partitions PEBS pages into weighted buckets by TLB coverage.

        Iterates over pages sorted by `TLB_COVERAGE` (descending, or
        ascending if `start_from_tail`) and assigns each page to the bucket
        whose remaining capacity best fits the page's weight, so that each
        bucket's total assigned weight approaches its target in
        `buckets_weights`. Used by `moselect_initial_samples` to build the
        static "group" layouts.

        Args:
            buckets_weights: List of target TLB-coverage percentages, one per
                bucket; consumed in place (decremented as pages are
                assigned).
            start_from_tail: If True, pages are considered from the lowest
                to the highest TLB coverage instead of the default
                highest-to-lowest order.
            fill_min_buckets_first: If True, ties are broken by preferring
                the bucket with the least remaining capacity that can still
                fit the page (best-fit); if False, the bucket with the most
                remaining capacity is preferred (worst-fit).

        Returns:
            list[list[int]]: One list of PEBS page numbers per bucket, in
            the same order as `buckets_weights`.
        """
        assert self.pebs_df is not None

        group_size = len(buckets_weights)
        group = [ [] for _ in range(group_size) ]
        df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=start_from_tail)

        threshold = 2
        i = 0
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            selected_weight = None
            selected_index = None
            completed_buckets = 0
            # count completed buckets and find bucket with minimal remaining
            # space to fill, i.e., we prefer to place current page in the
            # bicket that has the lowest remaining weight/space
            for i in range(group_size):
                if buckets_weights[i] <= 0:
                    completed_buckets += 1
                elif buckets_weights[i] >= weight - threshold:
                    if selected_index is None:
                        selected_index = i
                        selected_weight = buckets_weights[i]
                    elif fill_min_buckets_first and buckets_weights[i] < selected_weight:
                        selected_index = i
                        selected_weight = buckets_weights[i]
                    elif not fill_min_buckets_first and buckets_weights[i] > selected_weight:
                        selected_index = i
                        selected_weight = buckets_weights[i]
            if completed_buckets == group_size:
                break
            # if there is a bucket that has a capacity of current page, add it
            if selected_index is not None:
                group[selected_index].append(page)
                buckets_weights[selected_index] -= weight
        return group

    def moselect_initial_samples(self):
        """Builds the MosSelect-style static-profiling initial layouts.

        Partitions PEBS pages into three weighted buckets (targeting ~56%,
        ~28%, and ~14% TLB coverage respectively, i.e. successive halvings)
        via `fill_buckets`, then generates one candidate layout per subset of
        these three buckets (``2**3 = 8`` layouts total, including the empty
        and full subsets), so that combinations of the coarse coverage tiers
        are evaluated directly by the benchmark before BO takes over.

        Returns:
            list[list[int]]: One list of PEBS page numbers per candidate
            layout (up to 8 layouts).
        """
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fill_buckets(buckets_weights)
        mem_layouts = []
        # create eight layouts as all subgroups of these three group layouts
        for subset_size in range(len(group)+1):
            for subset in itertools.combinations(group, subset_size):
                subset_pages = list(itertools.chain(*subset))
                mem_layouts.append(subset_pages)
        return mem_layouts

    def chebyshev_tlb_misses_initial_samples(self, num_samples, min_misses, max_misses):
        """Builds initial layouts targeting Chebyshev-spaced STLB-miss counts.

        Chebyshev (Gauss-Chebyshev) nodes are denser near the interval
        endpoints, which is useful here because the relationship between TLB
        coverage and layout size is typically most sensitive at the extremes
        (very few or almost all hugepages). Each sampled target miss count is
        translated into a concrete layout via `generate_layout_from_pebs`
        (using the PEBS coverage that is expected to yield that miss count).

        Args:
            num_samples: Number of Chebyshev nodes (and thus layouts) to
                generate.
            min_misses: Lower bound of the target STLB-miss range (typically
                the all-hugepages layout's measured misses).
            max_misses: Upper bound of the target STLB-miss range (typically
                the no-hugepages layout's measured misses).

        Returns:
            list[list[int]]: One list of PEBS page numbers per generated
            layout.
        """
        chebyshev_dist = (chebgauss(num_samples)[0] + np.ones(num_samples)) * 0.5
        range_misses = max_misses - min_misses
        samples_misses = chebyshev_dist * range_misses + min_misses
        samples_misses = samples_misses.astype(np.uint64)

        mem_layouts = []
        for w in samples_misses:
            layout = self.generate_layout_from_pebs(w)
            mem_layouts.append(layout)

        return mem_layouts

    def chebyshev_initial_samples(self, num_samples):
        '''
        Generate initial samples for Bayesian optimization using
        Chebyshev distribution with discrete integer dimensions.
        Use roots_chebyt to obtain the Chebyshev nodes,
        scales the values to match the desired range,
        and rounds them to the nearest integer to align
        with the Integer dimension.
        '''
        chebyshev_dist = (chebgauss(num_samples)[0] + np.ones(num_samples)) * 0.5
        chebyshev_dist = chebyshev_dist.reshape((num_samples, 1))
        dimensions_space = np.full((1, self.num_dimensions), fill_value=self.dimension_max_val, dtype=np.int64)
        dimensions_space[0,-1] = self.last_dimension_max_val

        samples = chebyshev_dist * dimensions_space
        samples = np.rint(samples).astype(np.int64)

        decompressed_samples = [self.decompress_memory_layout(s) for s in samples]
        return decompressed_samples

    def generate_random_layout(self):
        """Generates one uniformly random hugepage layout.

        Each of the `num_default_hugepages` (base, i.e. un-grouped 2MB)
        hugepages is independently included with probability 0.5.

        Returns:
            list[int]: The base hugepage indices selected for this layout.
        """
        mem_layout = []
        random_mem_layout = np.random.randint(2, size=self.num_default_hugepages)
        for i in range(len(random_mem_layout)):
            if random_mem_layout[i] == 1:
                mem_layout.append(i)
        return mem_layout

    def random_initial_samples(self, num_initial_layouts):
        """Generates several independent random layouts.

        Args:
            num_initial_layouts: Number of random layouts to generate.

        Returns:
            list[list[int]]: One randomly generated layout (see
            `generate_random_layout`) per requested sample.
        """
        mem_layouts = []
        for i in range(num_initial_layouts):
            random_mem_layout = self.generate_random_layout()
            mem_layouts.append(random_mem_layout)
        return mem_layouts

    def base_mem_layouts(self):
        """Returns the two trivial baseline layouts: no and all hugepages.

        These correspond to the extremes of the search space and are used as
        the default (``base``) initialization strategy, giving `gp_minimize`
        both boundary points of the objective before it starts exploring.

        Returns:
            list[list[int]]: ``[[], [0, 1, ..., num_default_hugepages - 1]]``.
        """
        base_pages_layout = []
        hugepages_layout = [i for i in range(self.num_default_hugepages)]
        mem_layouts = [base_pages_layout, hugepages_layout]
        return mem_layouts

    def get_previous_run_samples(self):
        """Reloads previously evaluated layouts, enabling experiment resumption.

        Reads the aggregated results file and, for every already-executed
        layout, reconstructs its compressed representation from the
        Mosalloc layout CSV on disk (via `Utils.load_layout_hugepages`). This
        lets `run` pick up an interrupted experiment (e.g., after a crash or
        a cluster job time limit) without recomputing or re-running layouts
        that were already measured, and keeps `last_layout_num` consistent.

        Returns:
            tuple[list[list[int]], list[float]]: ``(X0, Y0)`` -- the
            compressed layouts and their measured STLB-miss counts,
            respectively, in the order they were originally executed. Both
            lists are empty if no results exist yet.
        """
        X0 = []
        Y0 = []
        res_df = BayesianExperiment.collect_results(self.collect_reults_cmd, self.results_file)
        if res_df.empty:
            return X0, Y0
        for index, row in res_df.iterrows():
            layout_name = row['layout']
            mem_layout_pages = Utils.load_layout_hugepages(layout_name, self.exp_root_dir)
            tlb_misses = row['stlb_misses']
            compressed_mem_layout = self.compress_memory_layout(mem_layout_pages)
            X0.append(compressed_mem_layout)
            Y0.append(tlb_misses)
            self.last_layout_num += 1
        return X0, Y0

    def generate_initial_samples(self, num_initial_points, type):
        """Produces (and evaluates) the initial ``(X0, Y0)`` sample set for BO.

        First resumes any previously evaluated layouts (`get_previous_run_samples`),
        then dispatches to the requested initialization strategy to obtain the
        (possibly larger) target list of initial layouts, skipping layouts
        already covered by the resumed samples, and finally runs each
        remaining layout on the real benchmark via `run_workload` to obtain
        its measured objective value.

        Args:
            num_initial_points: Requested number of initial samples (exact
                meaning depends on `type`; e.g. ignored by ``base`` and
                ``moselect``, which have a fixed sample count).
            type: One of ``'base'``, ``'random'``, ``'our_random'``,
                ``'chebyshev'``, ``'chebyshev_misses'``, or ``'moselect'``
                (see the module docstring for a description of each).

        Returns:
            tuple[list[list[int]], list[float]]: ``(X0, Y0)`` -- the
            compressed initial layouts and their measured objective values,
            ready to be passed as `gp_minimize`'s ``x0``/``y0``.

        Raises:
            ValueError: If `type` is not one of the supported strategies.
        """
        X0, Y0 = self.get_previous_run_samples()
        num_prev_samples = len(X0) if X0 else 0

        if type == 'base':
            mem_layouts = self.base_mem_layouts()
        elif type == 'random':
            mem_layouts = []
        elif type == 'our_random':
            mem_layouts = self.random_initial_samples(num_initial_points)
        elif type == 'chebyshev':
            mem_layouts = self.chebyshev_initial_samples(num_initial_points)
        elif type == 'chebyshev_misses':
            X0, Y0 = self.generate_initial_samples(2, 'base')
            mem_layouts = self.chebyshev_tlb_misses_initial_samples(8, Y0[1], Y0[0])
        elif type == 'moselect':
            mem_layouts = self.moselect_initial_samples()
        else:
            raise ValueError(f'Invalid initialization type to generate initial samples: {type}')
        for i, mem_layout in enumerate(mem_layouts):
            if i < num_prev_samples:
                self.last_layout_num += 1
                continue
            print(f'** Producing initial sample #{i} using a memory layout with {len(mem_layout)*self.hugepages_in_compressed_hugepage} (x2MB) hugepages')
            compressed_mem_layout = self.compress_memory_layout(mem_layout)
            X0.append(compressed_mem_layout)
            self.last_layout_num += 1
            layout_name = f'layout{self.last_layout_num}'
            tlb_misses = self.run_workload(compressed_mem_layout, layout_name)
            Y0.append(tlb_misses) # evaluate the objective function for each sample
        return X0, Y0

    def run_workload(self, compressed_mem_layout, layout_name):
        """Materializes, executes, and measures a single candidate layout.

        Decompresses `compressed_mem_layout` back into hugepage indices,
        writes the corresponding Mosalloc configuration CSV
        (`Utils.write_layout`), runs the benchmark under that layout via
        `run_experiment_cmd`, and finally reads back the measured STLB-miss
        count for this layout.

        Args:
            compressed_mem_layout: A point ``x`` in the compressed search
                space, as produced by `compress_memory_layout`.
            layout_name: Unique name for this layout (e.g. ``'layout7'``),
                used both for the Mosalloc config file and the benchmark's
                output directory.

        Returns:
            float: The measured STLB misses for this layout -- this is the
            objective value that `gp_minimize` seeks to minimize.

        Raises:
            RuntimeError: If running the benchmark (`run_experiment_cmd`)
                exits with a non-zero status.
        """
        mem_layout = self.decompress_memory_layout(compressed_mem_layout)
        Utils.write_layout(layout_name, mem_layout, self.exp_root_dir, self.brk_footprint, self.mmap_footprint)

        print('--------------------------------------')
        print(f'** Running {layout_name} with {len(mem_layout)} hugepages')
        out_dir = f'{self.exp_root_dir}/{layout_name}'
        run_bayesian_cmd = f'{self.run_experiment_cmd} {layout_name}'
        ret_code = BayesianExperiment.run_command(run_bayesian_cmd, out_dir)
        if ret_code != 0:
            raise RuntimeError(f'Error: running {layout_name} failed with error code: {ret_code}')
        tlb_misses, runtime = self.get_layout_results(layout_name)
        print(f'\tResults: runtime={runtime/1e9:.2f} Billion cycles , stlb-misses={tlb_misses/1e9:.2f} Billions')
        print('--------------------------------------')
        return tlb_misses

    # Define the objective function using named arguments and the use_named_args decorator
    # @use_named_args(self.dimensions)
    def objective_function(self, mem_layout):
        """The objective function minimized by `skopt.gp_minimize`.

        Allocates the next sequential layout name and delegates to
        `run_workload` to execute and measure it.

        Args:
            mem_layout: A point ``x`` in the compressed search space, as
                proposed by `gp_minimize` for the next evaluation.

        Returns:
            float: The measured STLB misses for `mem_layout` (lower is
            better).
        """
        # mem_layout = [params[f'mem_region_{i}'] for i in range(self.num_dimensions)]
        self.last_layout_num += 1
        layout_name = f'layout{self.last_layout_num}'
        return self.run_workload(mem_layout, layout_name)

    def run(self, initial_points=10, initialization_type='base'):
        """Runs the full Bayesian Optimization experiment to completion.

        Generates and evaluates the initial samples (`generate_initial_samples`),
        then, if fewer than `self.num_layouts` layouts have been evaluated so
        far, calls `skopt.gp_minimize` with the Expected-Improvement (``EI``)
        acquisition function to iteratively propose and evaluate the
        remaining layouts via `objective_function`, until `self.num_layouts`
        total evaluations have been performed.

        Args:
            initial_points: Requested number of initial samples; forwarded to
                `generate_initial_samples` (meaning depends on
                `initialization_type`).
            initialization_type: The initial-sample strategy to use; see
                `generate_initial_samples` and the module docstring.

        Side Effects:
            Executes the benchmark once per newly evaluated layout (each a
            real, potentially expensive run) and prints a summary of the
            final `skopt` optimization result.
        """
        # Define the initial data samples (X and Y pairs) for Bayesian optimization
        X0, Y0 = self.generate_initial_samples(initial_points, initialization_type)

        num_layouts = max(0, (self.num_layouts - len(X0)))
        if num_layouts == 0:
            print('================================================')
            print(f'No more layouts to run for the experiment:\n{self.exp_root_dir}')
            print('================================================')
            return

        # Perform Bayesian optimization with the initial data samples
        result = gp_minimize(self.objective_function,  # the objective function to minimize
                            dimensions=self.dimensions,  # the search space
                            acq_func='EI',  # the acquisition function
                            n_calls=num_layouts,  # the number of evaluations of f including at x0
                            x0=X0,  # the initial data samples
                            y0=Y0)  # the initial data sample evaluations

        print('================================================')
        print(f'Finished running Bayesian Optimization process for:\n{self.exp_root_dir}')
        print("result:", result)
        print('================================================')
        # print("Best TLB misses:", result.fun)
        # compressed_best_layout = [int(x) for x in result.x]
        # print("Best memory layout (compressed):", compressed_best_layout)
        # decompressed_best_layout = self.decompress_memory_layout(compressed_best_layout)
        # print(f"Best memory layout ({len(decompressed_best_layout)} items):")
        # if len(decompressed_best_layout) <= 20:
        #     print(decompressed_best_layout)
        # else:
        #     print(decompressed_best_layout[:10], '...', decompressed_best_layout[-10:])

import argparse
def parseArguments():
    """Parses command-line arguments for the Bayesian optimization experiment.

    See the module docstring for a full example invocation, and
    `experiments/bayesian_optimization/module.mk` for how the Makefile
    pipeline supplies these arguments.

    Returns:
        argparse.Namespace: The parsed arguments (memory_footprint,
        pebs_mem_bins, exp_root_dir, results_file, collect_reults_cmd,
        run_experiment_cmd, num_layouts, initialization_method, debug).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
    parser.add_argument('-p', '--pebs_mem_bins', default=None)
    parser.add_argument('-e', '--exp_root_dir', required=True)
    parser.add_argument('-r', '--results_file', required=True)
    parser.add_argument('-c', '--collect_reults_cmd', required=True)
    parser.add_argument('-x', '--run_experiment_cmd', required=True)
    parser.add_argument('-n', '--num_layouts', required=True, type=int)
    parser.add_argument('-i', '--initialization_method', choices=['base', 'random', 'chebyshev', 'chebyshev_misses', 'moselect'], default='base')
    parser.add_argument('-d', '--debug', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()

    # profiler = cProfile.Profile()
    # profiler.enable()
    exp = BayesianExperiment(args.memory_footprint, args.pebs_mem_bins,
                             args.collect_reults_cmd, args.results_file,
                             args.run_experiment_cmd, args.exp_root_dir,
                             args.num_layouts)
    exp.run(initialization_type=args.initialization_method)
    # profiler.disable()
    # profiler.dump_stats('profile_results.prof')
    # profiler.print_stats()
