#! /usr/bin/env python3

import sys, os

def predictAlamRuntime(B, walk_cycles):
    '''
    runtime = PAGE_WALK_LATENCY + B
    where:
    B = runtime[2mb] - PAGE_WALK_LATENCY
    '''
    y_pred = walk_cycles + B
    return y_pred

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-C', '--walk_cycles', type=float, required=True)
parser.add_argument('-f', '--coeffs_file', default='../analysis/linear_models_coeffs/coeffs.csv')
args = parser.parse_args()

import pandas as pd
coeffs = pd.read_csv(args.coeffs_file)
alam_runtime = predictAlamRuntime(coeffs['alam_B'][0], args.walk_cycles)

print('Alam model prediction:')
print('------------------------')
print(f'walk-cycles = {args.walk_cycles} --> runtime = {alam_runtime}')
