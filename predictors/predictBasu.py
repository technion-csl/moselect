#! /usr/bin/env python3

import sys, os

def predictBasuRuntime(A, B, tlb_misses):
    '''
    runtime = A * TLB_MISSES + B
    where:
    A = PAGE_WALK_LATENCY / TLB_MISSES
    B = runtime[4kb] - PAGE_WALK_LATENCY
    '''
    y_pred = A * tlb_misses + B
    return y_pred

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-M', '--tlb_misses', type=float, required=True)
parser.add_argument('-f', '--coeffs_file', default='../analysis/linear_models_coeffs/coeffs.csv')
args = parser.parse_args()

import pandas as pd
coeffs = pd.read_csv(args.coeffs_file)
basu_runtime = predictBasuRuntime(coeffs['basu_A'][0], coeffs['basu_B'][0], args.tlb_misses)

print('Basu model prediction:')
print('------------------------')
print(f'tlb-misses = {args.tlb_misses} --> runtime = {basu_runtime}')
