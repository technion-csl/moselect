#! /usr/bin/env python3

import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from performance_statistics import PerformanceStatistics

def loadDataframe(mean_file):
    mean_ps = PerformanceStatistics(mean_file, index_col='layout')
    mean_df = mean_ps.getDataFrame()
    mean_df['walk_cycles'] = mean_ps.getWalkDuration()
    mean_df['cpu-cycles'] = mean_ps.getRuntime()
    df = mean_df[['walk_cycles', 'cpu-cycles']]

    return df

def compareThpToLibhugetlbfs(df):
    thp_runtime = df.loc['2mb_thp']['cpu-cycles']
    libhugetlbfs_runtime = df.loc['2mb_libhugetlbfs']['cpu-cycles']
    thp_vs_libhugetlbfs = (libhugetlbfs_runtime - thp_runtime) / thp_runtime
    print('THP vs. libhugetlbfs difference: ',
          '{:.1%}'.format(thp_vs_libhugetlbfs))

def calculatePhamError(df):
    B = df.loc['2mb_libhugetlbfs']['cpu-cycles'] - \
            df.loc['2mb_libhugetlbfs']['walk_cycles']
    predicted_runtime = df.loc['1gb_libhugetlbfs']['walk_cycles'] + B
    true_runtime = df.loc['1gb_libhugetlbfs']['cpu-cycles']
    error = (predicted_runtime - true_runtime) / true_runtime
    print('Pham error: ', '{:.1%}'.format(error))

def calculateSwiftError(df):
    ideal_runtime = df.loc['2mb_thp']['cpu-cycles'] - df.loc['2mb_thp']['walk_cycles']
    min_walk_cycles = df.loc['1gb_libhugetlbfs']['walk_cycles']
    if min_walk_cycles < 0.01 * ideal_runtime:
        true_ideal_runtime = df.loc['1gb_libhugetlbfs']['cpu-cycles']
        error = (ideal_runtime - true_ideal_runtime) / true_ideal_runtime
        print('Swift error: ', '{:.1%}'.format(error))
    else:
        print('Swift error: unavailable')

def calculateYanivError(df):
    delta_y = df.loc['4kb']['cpu-cycles'] - df.loc['2mb_thp']['cpu-cycles']
    delta_x = df.loc['4kb']['walk_cycles'] - df.loc['2mb_thp']['walk_cycles']
    A = delta_y / delta_x
    B = df.loc['4kb']['cpu-cycles'] - A * df.loc['4kb']['walk_cycles']

    walk_cycles_1gb = df.loc['1gb_libhugetlbfs']['walk_cycles']
    predicted_runtime_1gb = A * walk_cycles_1gb + B
    true_runtime_1gb = df.loc['1gb_libhugetlbfs']['cpu-cycles']
    error = (predicted_runtime_1gb - true_runtime_1gb) / true_runtime_1gb
    print('Yaniv error: ', '{:.1%}'.format(error), '(slope={:.2f})'.format(A))
    return walk_cycles_1gb, predicted_runtime_1gb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mean_file', default='mean.csv')
parser.add_argument('-o', '--output',  default='summary.csv', required=True)
args = parser.parse_args()

df = loadDataframe(args.mean_file)
compareThpToLibhugetlbfs(df)
calculatePhamError(df)
calculateSwiftError(df)
walk_cycles_1gb, predicted_runtime_1gb = calculateYanivError(df)

# normalize to billions of cycles
summary_df = df.loc[['4kb', '2mb_thp', '1gb_libhugetlbfs']]
prediction_df = pd.DataFrame([[walk_cycles_1gb, predicted_runtime_1gb]],
        columns=summary_df.columns, index=['prediction'])
summary_df = summary_df.append(prediction_df)
summary_df['walk_cycles'] = 1e-9 * summary_df['walk_cycles']
summary_df['cpu-cycles'] = 1e-9 * summary_df['cpu-cycles']

summary_df.to_csv(args.output)

