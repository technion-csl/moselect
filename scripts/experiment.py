#! /usr/bin/env python3

import pandas as pd
import numpy as np

def readSingleFile(file_name, metrics_column=0, stats_column=1):
    try:
        metrics, stats = np.loadtxt(file_name, delimiter=',', dtype=str,
                unpack=True, usecols=[metrics_column, stats_column])
        df = pd.DataFrame({'stats': stats}, index=metrics)
        df['stats'] = pd.to_numeric(df['stats'], errors='coerce')
    except IOError:
        return None
    except:
        raise ValueError('Could not read the CSV file: ' + file_name)
    return df

class Experiment:
    def __init__(self, layout, experiments_root):
        self._layout = layout
        self._experiments_root = experiments_root

    def collect(self, repeat):
        experiment_dir = self._experiments_root + '/' + \
        self._layout + '/repeat' + str(repeat)
        perf_file_name = experiment_dir + '/perf.out'
        perf_df = readSingleFile(perf_file_name, metrics_column=2,
                                                stats_column=0)
        time_file_name = experiment_dir + '/time.out'
        time_df = readSingleFile(time_file_name)
        df = pd.concat([perf_df, time_df])
        df = pd.concat([perf_df, time_df])
        df = df.transpose()
        return df

    def __repr__(self):
        return 'experiment with ' + str(self.__dict__)

