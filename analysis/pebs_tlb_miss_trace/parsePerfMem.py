#! /usr/bin/env python3

import sys
import pandas as pd
import numpy as np

def writeDataframeToCsv(df, file_name):
    df.to_csv(file_name, na_rep='NaN')

def getAddress(strAddr):
    try:
        return int(strAddr, 16)
    except Exception:
        return 0

def getPageNumber(addr, base_addr):
    return str((getAddress(addr) - base_addr) >> 21)

def applyRanges(df, pid, anon_start, anon_end, brk_start, brk_end, file_start, file_end):
    df['ADDR'] = df['ADDR'].apply(
            lambda x: 'anon_page:' + getPageNumber(x, anon_start)
            if getAddress(x) >= anon_start and getAddress(x) < anon_end else x)
    df['ADDR'] = df['ADDR'].apply(
            lambda x: 'brk_page:' + getPageNumber(x, brk_start)
            if getAddress(x) >= brk_start and getAddress(x) < brk_end else x)
    df['ADDR'] = df['ADDR'].apply(
            lambda x: 'file_page:' + getPageNumber(x, file_start)
            if getAddress(x) >= file_start and getAddress(x) < file_end else x)


import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pools_range_file', required=True,
                    help='file that contains the pools addresses range')
parser.add_argument('-i', '--input_file', type=argparse.FileType('r'), default=sys.stdin,
                    help='input file that contains "perf mem -D report" results')
parser.add_argument('-o', '--output_file', required=True,
                    help='the output file to write memory accesses according to input file')
args = parser.parse_args()

# the perf mem output is built with ';' delimiter and not simple comma ','
# because it may contain some C++ function signatures which contain commas
df = pd.read_csv(args.input_file, delimiter=';')
df.columns = ['PID', 'TID', 'IP', 'ADDR', 'LOCAL WEIGHT', 'DSRC', 'SYMBOL']
df['#ACCESSES'] = 1
df = df[['PID', 'ADDR','#ACCESSES']]
df['ADDR'] = df['ADDR'].astype(str)

pools_df=pd.read_csv(args.pools_range_file, index_col='pid')
pids=pools_df.index.values
for pid in pids:
    p=pools_df.loc[pid]
    applyRanges(df, pid,
            getAddress(p['anon-mmap-start']), getAddress(p['anon-mmap-end']),
            getAddress(p['brk-start']), getAddress(p['brk-end']),
            getAddress(p['file-mmap-start']), getAddress(p['file-mmap-end']))


df_g = df.groupby(['PID', 'ADDR'], sort=False).size().reset_index()
df_g.columns = df.columns

df.sort_values('#ACCESSES', ascending=False, inplace=True)
df_g.sort_values('#ACCESSES', ascending=False, inplace=True)

grouped_df_file = args.output_file + '.grouped'
writeDataframeToCsv(df, args.output_file)
writeDataframeToCsv(df_g, grouped_df_file)


