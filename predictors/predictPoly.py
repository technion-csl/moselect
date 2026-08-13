#! /usr/bin/env python3

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../analysis/mosmodel')
import my_models
import utility

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-C', '--walk_cycles', type=float, required=True)
parser.add_argument('-d', '--poly_degree', type=int, choices=[1,2,3], default=3)
parser.add_argument('-t', '--train_dataset', default='../analysis/mosmodel/train/mean.csv')
args = parser.parse_args()

import pandas as pd
#train_df = pd.read_csv(args.train_dataset)
train_df = utility.loadDataframe(args.train_dataset)

model = my_models.poly3
if args.poly_degree == 1:
    model = my_models.poly1
elif args.poly_degree == 2:
    model = my_models.poly2

runtime = my_models.predictRuntime(model, train_df, ['walk_cycles'], [[args.walk_cycles]])

print(str.format('Polynomial (degree=${degree}) model prediction:', degree=args.poly_degree))
print('------------------------')
print(f'walk-cycles = {args.walk_cycles} --> runtime = {runtime}')
