#! /usr/bin/env python3

def maxError(error_column):
    max_err = error_column.max()
    min_err = error_column.min()
    abs_max_err = max_err
    if abs(min_err) > abs(max_err):
        abs_max_err = min_err
    #max_err_str = '{:.1%}'.format(abs_max_err)
    #return max_err_str
    return abs_max_err

def relativeError(y_true, y_pred):
    return (y_pred-y_true)/y_true

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from performance_statistics import PerformanceStatistics
def loadDataframe(mean_file):
    mean_ps = PerformanceStatistics(mean_file)
    mean_df = mean_ps.getDataFrame()
    mean_df['cpu-cycles'] = mean_ps.getRuntime()
    mean_df['walk_cycles'] = mean_ps.getWalkDuration()
    mean_df['stlb_hits'] = mean_ps.getStlbHits()
    mean_df['stlb_misses'] = mean_ps.getStlbMisses()
    df = mean_df[['layout', 'walk_cycles', 'stlb_hits', 'stlb_misses', 'cpu-cycles']]

    important_columns = list(df.columns)
    important_columns.remove('layout')
    #df.drop_duplicates(inplace=True, subset=important_columns)
    #df.sort_values('cpu-cycles', inplace=True)
    df = df.drop_duplicates(subset=important_columns)
    df = df.sort_values('cpu-cycles')

    return df


