#! /usr/bin/env python3

import pandas as pd
import my_models
import random
import utility

def evaluateDanErrors(df, baseline_errors):
    i = 0
    while (True): # will eventually stop due to the time limit
        if i % 1000 == 0:
            print("Starting iteration", i, "...")
        train_df = df.sample(frac=0.25, random_state=i)
        i += 1
        test_df = df.loc[df.index.difference(train_df.index)]
        models = {'poly1': my_models.poly1,
                'poly2': my_models.poly2,
                'poly3': my_models.poly3,
                'mosmodel': my_models.mosmodel}
        features = {'poly1': ['walk_cycles'],
                'poly2': ['walk_cycles'],
                'poly3': ['walk_cycles'],
                'mosmodel': ['walk_cycles', 'stlb_misses', 'stlb_hits']}
        tolerance = {'poly1': 0.1,
                'poly2': 0.1,
                'poly3': 0.1,
                'mosmodel': 0.02}
        errors = {}
        good_sample = True
        for m in model_list:
            error_list = my_models.calculateModelError(models[m], train_df, test_df, features[m])
            errors[m] = utility.maxError(error_list)
            if (errors[m] < baseline_errors[m] - tolerance[m]) or \
                    (errors[m] > baseline_errors[m] + tolerance[m]):
                        good_sample = False
        if good_sample:
            print("Found a good sample!")
            print("test set = ", test_df.index)
            print("baseline_errors = ", baseline_errors)
            print("calculated errors = ", errors)
            break # exit the while loop and finish

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-e', '--baseline_errors_file', required=True)
parser.add_argument('-b', '--benchmark', required=True)
parser.add_argument('-t', '--time_limit_seconds', type=int, default=10)
#parser.add_argument('-t', '--time_limit_seconds', default=3600) # 1 hour
args = parser.parse_args()

model_list = ['poly1', 'poly2', 'poly3', 'mosmodel']
baseline_errors_df = pd.read_csv(args.baseline_errors_file, index_col='benchmark')
baseline_errors = {m: baseline_errors_df.loc[args.benchmark, m] for m in model_list}

df = utility.loadDataframe(args.input)

import signal
from contextlib import contextmanager
class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds=0):
    def handler(signum, frame):
        raise TimeoutError('timed out after {} seconds'.format(seconds))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)

with timeout(seconds=args.time_limit_seconds):
    evaluateDanErrors(df, baseline_errors)

