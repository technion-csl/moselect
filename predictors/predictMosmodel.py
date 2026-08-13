#! /usr/bin/env python3

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../analysis/mosmodel')
import my_models
import utility

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-C', '--walk_cycles', type=float, required=True)
parser.add_argument('-H', '--tlb_hits', type=float, required=True)
parser.add_argument('-M', '--tlb_misses', type=float, required=True)
parser.add_argument('-t', '--train_dataset', default='../analysis/mosmodel/train/mean.csv')
args = parser.parse_args()

import pandas as pd
#train_df = pd.read_csv(args.train_dataset)
train_df = utility.loadDataframe(args.train_dataset)

runtime = my_models.predictRuntime(my_models.mosmodel, train_df,
        ['walk_cycles', 'stlb_misses', 'stlb_hits'],
        [[args.walk_cycles, args.tlb_misses, args.tlb_hits]])

print('Mosmodel prediction:')
print('--------------------')
print(f'C = {args.walk_cycles} , M = {args.tlb_misses} , H = {args.tlb_hits} ==> runtime = {runtime}')
