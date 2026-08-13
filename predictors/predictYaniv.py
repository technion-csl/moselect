#! /usr/bin/env python3

import sys, os

def predictYanivRuntime(A, B, walk_cycles):
    '''
    runtime = A * walk_cycles + B
    Where:
    delta_y = runtime[4kb] - runtime[2mb]
    delta_x = walk_cycles[4kb] - walk_cycles[4kb]
    A = delta_y / delta_x
    B = runtime[4kb] - A * walk_cycles[4kb]
    '''
    y_pred = A * walk_cycles + B
    return y_pred

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-C', '--walk_cycles', type=float, required=True)
parser.add_argument('-f', '--coeffs_file', default='../analysis/linear_models_coeffs/coeffs.csv')
args = parser.parse_args()

import pandas as pd
coeffs = pd.read_csv(args.coeffs_file)
yaniv_runtime = predictYanivRuntime(coeffs['yaniv_A'][0], coeffs['yaniv_B'][0], args.walk_cycles)

print('Yaniv model prediction:')
print('------------------------')
print(f'walk-cycles = {args.walk_cycles} --> runtime = {yaniv_runtime}')
