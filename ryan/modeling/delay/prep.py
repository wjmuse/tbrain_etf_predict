import os
import numpy as np
import pandas as pd
import datetime
import time

from etf_tools.experimental import prepare_named_etf
from etf_tools.parsers import etf_parser

from sklearn.utils import shuffle as sk_shuffle
from collections import OrderedDict
import pickle

from .hypers import (
    cont_feature_cols,
    cate_feature_cols,
    label_cols,
    window,
    src_dir
)


filepath = os.path.dirname(os.path.abspath(__file__))



def naming_by_date(df_gen):
    for df in df_gen:
        yr_month = df.date.values[0] // 100
        yield f'{df.date.values[-1]}'


def build_dataset(code, save_to=None):
    sample_file = os.path.join(src_dir, f'{code}.csv')
    X_all, y_all, name_all = prepare_named_etf(sample_file, cont_feature_cols + cate_feature_cols, label_cols, window=window,
                                               etf_prep=lambda df: df.drop_duplicates().fillna(np.nan),
                                               feature_trans=lambda dfx, dfy: dfx.values, label_trans=lambda dfx, dfy: dfy.T.values,
                                               naming=naming_by_date)
    X_all = np.asarray(X_all)
    y_all = np.asarray(y_all)

    save_to = save_to or os.path.join(filepath, 'data', f'{code}.pkl')
    save_dir = os.path.dirname(save_to)
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump((X_all, y_all, name_all), open(save_to, 'wb'))


def draw_by_name(code, name, load_from=None):
    load_from = load_from or os.path.join(filepath, 'data', f'{code}.pkl')
    X_all, y_all, name_all = pickle.load(open(os.path.join(filepath, 'data', f'{code}.pkl'), 'rb'))
    index = name_all.index(name)
    return (X_all[:index, :], y_all[:index, :]), (X_all[index, :], y_all[index, :])


if __name__ == '__main__':
    args = etf_parser.parse_args()
    code = args.code
    overwrite = args.overwrite
    save_to = os.path.join(filepath, 'data', f'{code}.pkl')

    if (not os.path.isfile(save_to)) or overwrite:
        print(f'Generate the trainingset for {code}')
        build_dataset(code)
        print(f'Save the trainingset to {save_to}')

    datemark = args.datemark
    print(f'Load one example by the datemark: {datemark}')

    (_, _), (X_val, y_val) = draw_by_name(code, datemark, load_from=save_to)
    print(f'X: shape: {X_val.shape}')
    print(X_val)
    print(f'y: shape: {y_val.shape}')
    print(y_val)
