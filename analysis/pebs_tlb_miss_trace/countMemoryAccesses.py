#! /usr/bin/env python3

import sys
import pandas as pd
import numpy as np

def writeDataframeToCsv(df, file_name):
    df.to_csv(file_name, na_rep='NaN')

def hex_int(x):
    try:
        return int(x, 16)
    except Exception:
        return 0

import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pools_range_file', required=True,
                    help='file that contains the pools addresses range')
parser.add_argument('-i', '--input_file', type=argparse.FileType('r'), default=sys.stdin,
                    help='input file that contains "perf mem -D report" results')
parser.add_argument('-o', '--output_file', type=argparse.FileType('w'), default=sys.stdout,
                    help='the output file to write memory accesses according to input file')
args = parser.parse_args()

df = pd.read_csv(args.input_file, delimiter=';')
df.columns = ['PID', 'TID', 'IP', 'ADDR', 'LOCAL WEIGHT', 'DSRC', 'SYMBOL']
df = df[['PID', 'ADDR','LOCAL WEIGHT']]
df_addr = df.ADDR.apply(lambda x: hex_int(x))

pools_df=pd.read_csv(args.pools_range_file, index_col='pid')
pids=pools_df.index.values
max_accesses=0
for index, row in pools_df.iterrows():
    ranges=[hex_int(row['brk-start']), hex_int(row['brk-end'])-1,
            hex_int(row['anon-mmap-start']), hex_int(row['anon-mmap-end'])-1]
    res=df_addr.groupby(
            pd.cut(df_addr, ranges, right=True,include_lowest=True,\
                    duplicates='drop', \
                    labels=['brk','none','anon-mmap'])
            ).count()
    args.output_file.write('PID: ' + str(index) + '\n')
    args.output_file.write(res.to_string())
    args.output_file.write('\n--------------\n')
    max_accesses = max(res.sum(), max_accesses)

print('*******************************\n')
print('# memory accesses: ',max_accesses)
print('*******************************\n')


