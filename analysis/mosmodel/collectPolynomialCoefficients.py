#! /usr/bin/env python3

import sys
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='analysis/linear_models/',
                    help='the directory containing polynomial coefficients in a tree '+\
                    'structure of <benchmark>/poly.csv')
parser.add_argument('-b', '--benchmarks', required=True,
                    help='a comma-separated list of benchmarks')
parser.add_argument('-o', '--output', required=True,
                    help='output file to save the full coefficients table')
args = parser.parse_args()

try:
    benchmark_list = args.benchmarks.strip().split(',')
except KeyError:
    sys.exit('Error: could not parse the --benchmarks argument')

poly_df = pd.DataFrame()

for benchmark in benchmark_list:
    poly_file = args.root + '/' + benchmark + '/poly.csv'
    df = pd.read_csv(poly_file, index_col=False)
    poly_df = poly_df.append(df)

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from shortenBenchmarkName import shortenBenchmarkName
benchmark_list = [shortenBenchmarkName(b) for b in benchmark_list]

poly_df['benchmark'] = benchmark_list
poly_df.set_index('benchmark', inplace=True)
poly_df.to_csv(args.output, index_label='benchmark')

