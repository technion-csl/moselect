#!/usr/bin/env python3
"""CLI entry point for the MosSelect adaptive layout generator.

This script is invoked once per layout by `experiments/moselect/module.mk`
(one invocation == one new Mosalloc layout configuration file written under
``<exp_dir>/layouts/``). Each invocation:

1. Loads the benchmark's brk/mmap pool footprints (`--memory_footprint`).
2. Loads whatever layout results have been measured so far
   (`--results_file`), if any.
3. Loads and normalizes the one-time PEBS TLB-access sample
   (`--pebs_mem_bins`) into per-page coverage percentages.
4. Builds a `LayoutGenerator` and asks it to produce the layout named by
   `--layout` (e.g. ``layout7``), persisting it and all algorithm state to
   ``--exp_dir``.

See `layout_generator.py` for the full two-phase MosSelect algorithm this
delegates to, and the "MosSelect Layout Selector" section of the top-level
README for a higher-level description.
"""
import argparse

import pandas as pd

from layout_generator import LayoutGenerator, LayoutGeneratorUtils


def parseArguments():
    """Parses command-line arguments for a single layout-generation invocation.

    Returns:
        argparse.Namespace: Parsed arguments, including `memory_footprint`,
        `pebs_mem_bins`, `max_gap`, `max_budget`, `layout`, `exp_dir`,
        `results_file`, and `debug`.
    """
    parser = argparse.ArgumentParser(
        description='Generate the next MosSelect hugepage layout to evaluate for one benchmark.')
    parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt',
                         help='CSV file with the benchmark brk/mmap pool sizes (columns: brk-max, anon-mmap-max).')
    parser.add_argument('-p', '--pebs_mem_bins', default='mem_bins_2mb.csv',
                         help='CSV file with the raw PEBS memory-access samples for the brk pool.')
    parser.add_argument('-g', '--max_gap', type=int, default=4,
                         help='Maximum acceptable real-coverage gap (percentage points) between adjacent measured layouts.')
    parser.add_argument('-b', '--max_budget', type=int, default=50,
                         help='Total number of layouts allowed for this experiment.')
    parser.add_argument('-l', '--layout', required=True,
                         help='Name of the layout to generate in this invocation, e.g. "layout7".')
    parser.add_argument('-e', '--exp_dir', required=True,
                         help='Root experiment directory (layouts/, logs, and results live under here).')
    parser.add_argument('-r', '--results_file', required=True,
                         help='CSV file with performance results measured so far for this experiment.')
    parser.add_argument('-d', '--debug', action='store_true',
                         help='Dry run: print what would be generated without writing any files.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArguments()

    # read memory-footprints
    footprint_df = pd.read_csv(args.memory_footprint)
    mmap_footprint = footprint_df['anon-mmap-max'][0]
    brk_footprint = footprint_df['brk-max'][0]

    LayoutGeneratorUtils.setPoolsFootprints(brk_footprint, mmap_footprint)

    results_df = LayoutGeneratorUtils.loadDataframe(args.results_file)

    pebs_df = LayoutGeneratorUtils.normalizePebsAccesses(args.pebs_mem_bins)

    layout_generator = LayoutGenerator(pebs_df, results_df, args.layout, args.exp_dir, args.max_gap, args.max_budget, args.debug)
    layout_generator.generateLayout()
