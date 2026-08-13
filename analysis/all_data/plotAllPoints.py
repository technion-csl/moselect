#! /usr/bin/env python3

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from performance_statistics import PerformanceStatistics
def loadDataframe(mean_file, output):
    mean_ps = PerformanceStatistics(mean_file)
    mean_df = mean_ps.getDataFrame()
    mean_df['cpu-cycles'] = mean_ps.getRuntime()
    mean_df['walk_cycles'] = mean_ps.getWalkDuration()
    mean_df['stlb_hits'] = mean_ps.getStlbHits()
    mean_df['stlb_misses'] = mean_ps.getStlbMisses()
    df = mean_df[['layout', 'walk_cycles', 'stlb_hits', 'stlb_misses', 'cpu-cycles']]
    # drop duplicated rows
    important_columns = list(df.columns)
    important_columns.remove('layout')
    #df.drop_duplicates(inplace=True, subset=important_columns)
    df = df.drop_duplicates(subset=important_columns)
    df.to_csv(output)
    return df

import matplotlib.pyplot as plt
def plot(x_axis, x_label, output, equal_axis=True):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,3))
    ax.plot(train_df[x_axis], train_df[['cpu-cycles']], 'r+', label='train')
    ax.plot(test_df[x_axis], test_df[['cpu-cycles']], 'bx', label='test')

    plt.gca().yaxis.grid(True)
    if equal_axis:
        ax.axis('equal')
    ax.margins(x=0)
    ax.margins(y=0)

    # set x, y labels
    plt.xlabel(x_label)
    plt.ylabel('cpu cycles')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best')

    # save to pdf
    fig.savefig(output, bbox_inches='tight')
    #plt.show()
    plt.close('all')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_mean_file', default='train_mean.csv')
parser.add_argument('-v', '--test_mean_file', default='test_mean.csv')
parser.add_argument('-o', '--output',  default='all_points', required=True)
args = parser.parse_args()

#read mean files
train_df = loadDataframe(args.train_mean_file, args.output + 'train_scatter.csv')
test_df = loadDataframe(args.test_mean_file, args.output + 'test_scatter.csv')

plot('walk_cycles', 'table walk cycles', args.output + 'table_walks.pdf', True)
plot('stlb_misses', 'tlb misses', args.output + 'tlb_misses.pdf', False)
plot('stlb_hits', 'tlb hits', args.output + 'tlb_hits.pdf', False)

