#! /usr/bin/env python3

import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from performance_statistics import PerformanceStatistics

def loadDataframe(mean_file):
    mean_ps = PerformanceStatistics(mean_file)
    mean_df = mean_ps.getDataFrame()
    mean_df['walk_cycles'] = mean_ps.getWalkDuration()
    mean_df['cpu-cycles'] = mean_ps.getRuntime()
    mean_df['stlb_misses'] = mean_ps.getStlbMisses()
    mean_df['stlb_hits'] = mean_ps.getStlbHits()
    df = mean_df[['walk_cycles', 'stlb_misses', 'stlb_hits', 'cpu-cycles']]

    return df

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='mean.csv',
                    help='the mean file to be used for calculating the R-squares')
parser.add_argument('-o', '--output',  default='r_squares.csv')
args = parser.parse_args()

from sklearn.linear_model import LinearRegression
res_df = pd.DataFrame(columns=['C', 'M', 'H'])
df = loadDataframe(args.input)

linear_regressor = LinearRegression(fit_intercept=True)

linear_regressor.fit(df[['walk_cycles']], df['cpu-cycles'])
walk_cycles_r_squared = linear_regressor.score(df[['walk_cycles']], df['cpu-cycles'])

linear_regressor.fit(df[['stlb_misses']], df['cpu-cycles'])
tlb_misses_r_squared = linear_regressor.score(df[['stlb_misses']], df['cpu-cycles'])

linear_regressor.fit(df[['stlb_hits']], df['cpu-cycles'])
tlb_hits_r_squared = linear_regressor.score(df[['stlb_hits']], df['cpu-cycles'])

res_df.loc[0] = {
        'C' : walk_cycles_r_squared,
        'M' : tlb_misses_r_squared,
        'H' : tlb_hits_r_squared}

res_df.to_csv(args.output, float_format='%.3f')
