#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mosalloc_4k_mean', default='analysis/single_page_size/4kb_mosalloc/mean.csv')
parser.add_argument('-g', '--glibc_malloc_mean', default='analysis/general_metrics/glibc_malloc/mean.csv')
parser.add_argument('-b', '--benchmarks', required=True,
                    help='a comma-separated list of benchmarks')
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

import pandas as pd
mosalloc_df = pd.read_csv(args.mosalloc_4k_mean, index_col='benchmark')

try:
    benchmark_list = args.benchmarks.strip().split(',')
except KeyError:
    sys.exit('Error: could not parse the --benchmarks argument')

GB_IN_KB = 1024*1024
res = []
for benchmark in benchmark_list:
    mosalloc_footprint = mosalloc_df.loc[benchmark, 'max-resident-memory-kb']
    mosalloc_footprint /= GB_IN_KB
    mosalloc_runtime = int(mosalloc_df.loc[benchmark, 'seconds-elapsed'])

    glibc_mean_df = pd.read_csv(args.glibc_malloc_mean, index_col='benchmark')
    glibc_footprint = glibc_mean_df.loc[benchmark, 'max-resident-memory-kb']
    glibc_footprint /= GB_IN_KB
    glibc_runtime = int(glibc_mean_df.loc[benchmark, 'seconds-elapsed'])

    memory_overhead_gb = mosalloc_footprint - glibc_footprint
    memory_overhead = (mosalloc_footprint - glibc_footprint) / glibc_footprint
    runtime_overhead_sec = mosalloc_runtime - glibc_runtime
    runtime_overhead = (mosalloc_runtime - glibc_runtime) / glibc_runtime

    memory_overhead_str = "{0:.0f}%".format(memory_overhead * 100)
    runtime_overhead_str = "{0:.0f}%".format(runtime_overhead * 100)

    res.append([benchmark, mosalloc_footprint, glibc_footprint, \
            memory_overhead_gb, memory_overhead_str, \
            mosalloc_runtime, glibc_runtime, \
            runtime_overhead_sec, runtime_overhead_str])

res_df = pd.DataFrame(res, \
        columns=['benchmark', \
        'mosalloc-footprint [GB]', 'glibc-footprint [GB]', 'memory-overhead [GB]', 'memory-overhead [%]',
        'mosalloc-runtime [sec]', 'glibc-runtime [sec]', 'runtime-overhead [sec]', 'runtime-overhead [%]'])
res_df.to_csv(args.output, index=False, float_format='%.2f')

