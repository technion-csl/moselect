#! /usr/bin/env python3

import pandas as pd
import my_models
import utility

from sklearn.model_selection import KFold

def crossValidate(df, model, features):
    res = pd.Series()
    kf = KFold(n_splits=len(df), shuffle=True, random_state=0)
    for train_index, test_index in kf.split(df):
        error = my_models.calculateModelError(model, df.iloc[train_index], df.iloc[test_index], features)
        res = pd.concat([res, error])
    res_df = df.copy()
    res_df = res_df[['layout']]
    res_df['cv_error'] = res
    return res_df

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

#df = pd.read_csv(args.input, index_col='layout')
df = utility.loadDataframe(args.input)
res_df = df.copy()
res_df = res_df[['layout']]
cv_df = crossValidate(df, my_models.mosmodel, ['walk_cycles', 'stlb_misses', 'stlb_hits'])
res_df['mosmodel_error'] = cv_df['cv_error']
cv_df = crossValidate(df, my_models.poly3, ['walk_cycles'])
res_df['poly3_error'] = cv_df['cv_error']
cv_df = crossValidate(df, my_models.poly2, ['walk_cycles'])
res_df['poly2_error'] = cv_df['cv_error']
cv_df = crossValidate(df, my_models.poly1, ['walk_cycles'])
res_df['poly1_error'] = cv_df['cv_error']
res_df.to_csv(args.output)
