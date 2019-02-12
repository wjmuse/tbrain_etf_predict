import os
import numpy as np
import pandas as pd
import datetime
import time
from sklearn.utils import shuffle as sk_shuffle
from .callbacks import EtfCkptLogger
from etf_tools.parsers import etf_parser

from collections import OrderedDict
from .models import build_fn
from .prep import build_dataset, draw_by_name
from .hypers import (
    this_dir,
    n_experiment,
    epochs,
    ckpt_step,
    ckpt_skip,
    batchsize
)
import keras.backend as K
import gc

PJ = os.path.join
PDIR = os.path.dirname
PABS = os.path.abspath

metrics_path = PJ(this_dir, 'validation', 'metrics.csv')
ckpt_dir = PJ(this_dir, 'ckpt')
headers = ['code', 'datemark', 'n_exp', 'epoch', 'price_score', 'updown_score']
delimiter = ','


if __name__ == '__main__':
    args = etf_parser.parse_args()
    code = args.code
    datemark = args.datemark
    overwrite = args.overwrite
    keep_model = args.keep_model


    metrics_dir = PDIR(metrics_path)
    if (not os.path.isfile(metrics_path)) or overwrite:
        EtfCkptLogger.create_dirs(metrics_dir, exist_ok=True)
        EtfCkptLogger.create_csv(metrics_path, headers=headers, delimiter=delimiter)

    EtfCkptLogger.create_dirs(ckpt_dir, exist_ok=True)

    load_from = os.path.join(this_dir, 'data', f'{code}.pkl')

    (X_train, y_train), (X_val, y_val) = draw_by_name(code, datemark, load_from=load_from)
    
    for i in range(n_experiment):
        tr = sk_shuffle(*X_train, *y_train)
        X_tr, y_tr = tr[:3], tr[-2:]
        tags = OrderedDict(code=code, datemark=datemark, n_exp=(i+1))
        model = build_fn()
        csv_logger = EtfCkptLogger(model, tags=tags,
                                   x_val=X_val, y_val=y_val,
                                   keep_model=keep_model,
                                   metrics_path=metrics_path, ckpt_dir=ckpt_dir,
                                   skip=ckpt_skip, step=ckpt_step)

        history = model.fit(x=X_tr, y=y_tr,
                            epochs=epochs,
                            batch_size=batchsize,
                            callbacks=[csv_logger],
                            validation_data=(X_val, y_val))

        del history
        K.clear_session()
        gc.collect()

