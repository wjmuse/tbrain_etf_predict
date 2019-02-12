import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import datetime
import argparse
from etf_tools.parsers import etf_parser
from etf_tools.experimental import etf_overall_score
from .hypers import this_dir
from .models import build_fn
from .prep import build_dataset, draw_by_name


PJ = os.path.join
PDIR = os.path.dirname
PABS = os.path.abspath

headers = ['code', 'datemark', 'etf_score']
delimiter = ','
metrics_path = PJ(this_dir, 'evaluate', 'metrics.csv')


def create_csv(path, headers=[], delimiter=','):
    with open(path, 'w') as f:
        f.write(f'{delimiter.join(headers)}\n')


if __name__ == '__main__':
    args = etf_parser.parse_args()
    code = args.code
    datemark = args.datemark
    overwrite = args.overwrite

    metrics_dir = PDIR(metrics_path)
    if (not os.path.isfile(metrics_path)) or overwrite:
        os.makedirs(metrics_dir, exist_ok=True)
        create_csv(metrics_path, headers=headers, delimiter=delimiter)

    load_from = os.path.join(this_dir, 'data', f'{code}.pkl')

    (_, _), (X_val, y_val) = draw_by_name(code, datemark, load_from=load_from)

    model = build_fn()

    X_val, y_val = X_val.flatten(), y_val.flatten()

    y_pred = model.predict(X_val)

    updown_pred = (1 * (np.diff(X_val) >= 0)).astype(int)
    updown_actual =  (1 * (np.diff(np.concatenate((X_val[-1:], y_val))) >= 0)).astype(int)


    overall_score = etf_overall_score(price_actual=y_val, price_predict=y_pred, updown_actual=updown_actual, updown_predict=updown_pred)


    with open(metrics_path, 'a') as f:
        f.write(f'{delimiter.join([code, datemark, str(overall_score)])}\n')

    print(f'code: {code}')
    print(f'datemark: {datemark}')
    print(f'etf score: {str(overall_score)}')
    print()



