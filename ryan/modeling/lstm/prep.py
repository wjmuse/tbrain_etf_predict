import os
import numpy as np
import pandas as pd
import datetime
import time

from etf_tools.experimental import prepare_named_etf
from etf_tools.tech import trend
from etf_tools.parsers import etf_parser

from sklearn.utils import shuffle as sk_shuffle
from collections import OrderedDict
import pickle

from .hypers import (
    cont_feature_cols,
    cate_feature_cols,
    n_lastdays,
    label_cols,
    window,
    src_dir
)


filepath = os.path.dirname(os.path.abspath(__file__))


# def progressive_group_split(name_list):
#     group_set = dict()
#     group_list = []
#     idx = 0
#     for e in name_list:
#         if e not in group_set:
#             group_set[e] = idx
#             group_list.append(e)
#             idx += 1
# 
#     n_group = len(group_set)
#     groups = np.zeros((n_group, len(name_list))).astype(int)
#     
#     for i, e in enumerate(name_list):
#         groups[group_set[e], i] = 1
#     
#     for i in range(n_group):
#         pos_idx, = np.where(groups[i] == 1)
#         pos_min = min(pos_idx)
#         neg_idx, = np.where(groups[i, :pos_min] == 0)
#         yield group_list[i], neg_idx, pos_idx



# # Hyperparams
# feature_dim = len(cont_feature_cols + cate_feature_cols)
# num_classes = len(label_cols)
# timestep = window = 5
# batchsize = 128
# epochs = 100
# dropout = 0
# n_experiment = 10
# min_training_num = 120


def feature_prep(df):
    incomp_months = pd.Series(df.date.apply(lambda e: (e // 100) % 100))
    num_data = incomp_months.shape[0]

    compens = pd.Series(range(1, 13))
    comp_months = pd.concat([incomp_months, compens]).reset_index(drop=True)

    months = pd.get_dummies(comp_months, prefix='month').head(num_data)
    weeks = pd.get_dummies(pd.Series(df.date.apply(lambda e: 1 + (e % 100) // 7)), prefix='week')
    return pd.concat([months, weeks, df], axis=1)


def feature_trans(dfx, dfy, n_lastdays=n_lastdays):
    price = dfx.close
    mean = price.mean()
    std = price.std()

    conts = dfx[cont_feature_cols]
    norms = ((conts - conts.mean()) / conts.std()).fillna(0)

    lastdays = norms.close[-n_lastdays:]    # shape: (n_lastdays,)
    statistics = np.asarray([mean, std])    # shape: (2,)

    return [pd.concat([dfx[cate_feature_cols], norms], axis=1).values, lastdays.values, statistics]


def label_trans(dfx, dfy):
    price = dfx.close
    mean = price.mean()
    std = price.std()

    if std < 0.01:
        raise ValueError('The price values are too cloesd.')

    y_price = ((dfy.T - mean) / std).values
    y_updown = (1 * (np.diff(np.concatenate((price[-1:].values, dfy.values.flatten()))) >= 0)).astype(int)

    return [np.expand_dims(y_updown, axis=-1), y_price]


def naming_by_date(df_gen):
    for df in df_gen:
        yr_month = df.date.values[0] // 100
        yield f'{df.date.values[-1]}'


def build_dataset(code, save_to=None):
    sample_file = os.path.join(src_dir, f'{code}.csv')
    X_all, y_all, name_all = prepare_named_etf(sample_file, cont_feature_cols + cate_feature_cols, label_cols, window=window,
                                               etf_prep=lambda df: df.drop_duplicates().fillna(np.nan),
                                               feature_prep=feature_prep,
                                               feature_trans=feature_trans,
                                               label_trans=label_trans,
                                               naming=naming_by_date)


    save_to = save_to or os.path.join(filepath, 'data', f'{code}.pkl')
    save_dir = os.path.dirname(save_to)
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump((X_all, y_all, name_all), open(save_to, 'wb'))


def draw_by_name(code, name, load_from=None, truncate=5):
    load_from = load_from or os.path.join(filepath, 'data', f'{code}.pkl')
    X_all, y_all, name_all = pickle.load(open(load_from, 'rb'))
    index = name_all.index(name)

    X_tr, y_tr = X_all[:index-truncate], y_all[:index-truncate]

    x_main_seq = np.asarray([x[0] for x in X_tr])
    x_lastdays = np.asarray([x[1] for x in X_tr])
    x_statistics = np.asarray([x[2] for x in X_tr])

    y_updown = np.asarray([y[0] for y in y_tr])
    y_price = np.asarray([y[1] for y in y_tr])

    X_val = [np.expand_dims(x, axis=0) for x in X_all[index]]
    y_val = [np.expand_dims(y, axis=0) for y in y_all[index]]

    return ([x_main_seq, x_lastdays, x_statistics], [y_updown, y_price]), (X_val, y_val)


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
    print('X:')
    print(X_val)
    print('y:')
    print(y_val)


