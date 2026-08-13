#! /usr/bin/env python3

import sys, os

def predictPhamRuntime(B, tlb_hits, walk_cycles):
    '''
    runtime = 1xDTLB_HITS + 7xSTLB_HITS + PAGE_WALK_LATENCY + B
    where:
    (DTLB_HITS can be ignored)
    B = runtime[4kb] - (7xSTLB_HITS + PAGE_WALK_LATENCY)
    '''
    y_pred = 7 * tlb_hits + walk_cycles + B
    return y_pred

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-H', '--tlb_hits', type=float, required=True)
parser.add_argument('-C', '--walk_cycles', type=float, required=True)
parser.add_argument('-f', '--coeffs_file', default='../analysis/linear_models_coeffs/coeffs.csv')
args = parser.parse_args()

import pandas as pd
coeffs = pd.read_csv(args.coeffs_file)
pham_runtime = predictPhamRuntime(coeffs['pham_B'][0], args.tlb_hits, args.walk_cycles)

print('Pham model prediction:')
print('------------------------')
print(f'tlb-hits = {args.tlb_hits}, walk-cycles = {args.walk_cycles} --> runtime = {pham_runtime}')
