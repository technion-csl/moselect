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
    mean_df['stlb_hits'] = mean_ps.getStlbHits()
    df = mean_df[['layout', 'walk_cycles', 'stlb_hits', 'cpu-cycles']]

    return df

def calculatePhamRuntime(df, pham_C):
    '''
    Pham_runtime = 1xDTLB_HITS + 7xSTLB_HITS + PAGE_WALK_LATENCY + C
    DTLB_HITS can be ignored, therefore:
    '''
    df['pham_runtime'] = 7*df['stlb_hits'] + df['walk_cycles'] + pham_C
    return df

def calculateYanivRuntime(df, yaniv_A, yaniv_B):
    df['yaniv_runtime'] = yaniv_A * df['walk_cycles'] + yaniv_B
    return df

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mean_file', default='mean.csv')
parser.add_argument('-c', '--coeffs_file', default='linear_models_coeffs.csv')
parser.add_argument('-b', '--benchmark', required=True)
parser.add_argument('-o', '--output',  default='errors.csv', required=True)
args = parser.parse_args()

df = loadDataframe(args.mean_file)
coeffs = pd.read_csv(args.coeffs_file, index_col='benchmark')
df = calculatePhamRuntime(df, coeffs.loc[args.benchmark]['pham_C'])
df = calculateYanivRuntime(df, coeffs.loc[args.benchmark]['yaniv_A'],
        coeffs.loc[args.benchmark]['yaniv_B'])

x = df[['walk_cycles']]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,3))
ax.plot(x, df[['cpu-cycles']], 'r+', label='measurements')
ax.plot(x, df[['pham_runtime']], 'bs', label='pham model')
ax.plot(x, df[['yaniv_runtime']], 'g^', label='yaniv model')

plt.gca().yaxis.grid(True)
#ax.axis('equal')
ax.margins(x=0)
ax.margins(y=0)

# set x, y labels
plt.xlabel('table walk cycles')
plt.ylabel('cpu cycles')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='best')

# save to pdf
fig.savefig(args.output, bbox_inches='tight')
plt.show()
plt.close('all')

