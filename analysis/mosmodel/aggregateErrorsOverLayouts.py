#! /usr/bin/env python3

import sys
import pandas as pd
import utility
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--errors_file', required=True,
        help='the errors file')
parser.add_argument('-f', '--function', required=True, choices=['max', 'avg'],
        help='the errors file name')
parser.add_argument('-c', '--columns', required=False, default=None,
        help='columns list to be considered in the aggregation')
parser.add_argument('-o', '--output', required=True,
        help='average/max errors output file')
args = parser.parse_args()

if args.columns == None:
    models = ['basu', 'alam', 'pham', 'gandhi', 'yaniv', 'poly1', 'poly2', 'poly3', 'mosmodel']
else:
    models = args.columns.strip().split(',')

errors_cols = [m+'-'+args.function+'-error' for m in models]
res_df = pd.DataFrame(columns=errors_cols)

from scipy.stats.mstats import gmean
df = pd.read_csv(args.errors_file, index_col='layout')
for model in models:
    error_column = df[model+'_error']
    if args.function == 'max':
        error = utility.maxError(error_column)
    else:
        error = gmean(error_column.abs())
    res_df[model+'-'+args.function+'-error'] = error

res_df.to_csv(args.output)


