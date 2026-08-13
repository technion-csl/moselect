#! /usr/bin/env python3

def shortenBenchmarkName(benchmark_name):
    benchmark_name = benchmark_name.replace('_cpu20', '')
    benchmark_name = benchmark_name.replace('my_gups', 'gups')
    benchmark_name = benchmark_name.replace('sequential-', '')
    benchmark_name = benchmark_name.replace('unionized-', '')
    benchmark_name = benchmark_name.replace('graph500-2.1', 'graph500')
    if '/' in benchmark_name and '.' in benchmark_name:
        benchmark_name = benchmark_name[:benchmark_name.find('/')+1] + benchmark_name[benchmark_name.find('.')+1:]
    return benchmark_name



