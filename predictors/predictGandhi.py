#! /usr/bin/env python3

import sys, os

def predictGandhiRuntime(A, B, tlb_misses):
    '''
    runtime = A * stlb_misses + B
    Where:
    B = runtime[2mb] - walk_cycles[2mb]
    A = walk_cycles / stlb_misses
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
gandhi_runtime = predictGandhiRuntime(coeffs['gandhi_A'][0], coeffs['gandhi_B'][0], args.tlb_misses)

print('Gandhi model prediction:')
print('------------------------')
print(f'tlb-misses = {args.tlb_misses} --> runtime = {gandhi_runtime}')
