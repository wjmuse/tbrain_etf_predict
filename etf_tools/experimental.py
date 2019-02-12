import pandas as pd
import numpy as np
# from sklearn.metrics import accuracy_score
from .ml import frame_rolling


def price_accuracy_score(actual, predict, sample_weight: list=[0.1, 0.15, 0.2, 0.25, 0.3]) -> float:
    result = [w * ((a - abs(a - p)) / a) for a, p, w in zip(actual, predict, sample_weight)]
    return sum(result) / sum(sample_weight)


def updown_accuracy_score(actual, predict, sample_weight: list=[1, 1, 1, 1, 1]) -> float:
    result = [w * (a == p) for a, p, w in zip(actual, predict, sample_weight)]
    return sum(result) / sum(sample_weight)


def etf_overall_score(price_actual=None, price_predict=None, price_weight=[0.1, 0.15, 0.2, 0.25, 0.3],
                      updown_actual=None, updown_predict=None, updown_weight=[1, 1, 1, 1, 1]) -> float:
    price_ = 0.5 * price_accuracy_score(price_actual, price_predict, sample_weight=price_weight)
    updown_ = 0.5 * updown_accuracy_score(updown_actual, updown_predict, sample_weight=updown_weight)
    return updown_ + price_


def prepare_named_etf_gen(fpath, feature_cols, label_cols, window=5, step=1, skip=0,
                          etf_prep=lambda df: df.drop_duplicates().dropna(),
                          feature_prep=lambda df: df, label_prep=lambda df: df,
                          feature_trans=lambda dfx, dfy: dfx, label_trans=lambda dfx, dfy: dfy,
                          naming=None, read_csv_kwargs={}):
    """TODO.

    Pipeline sequence: read_csv --> etf_prep
                                --> (feature_prep, label_prep)
                                --> (select feature_cols, select label_cols)
                                --> (feature_trans, label_trans)
    Params:
        fpath: file path of the etf spreadsheet.
        feature_cols: all column names must in the dataframe after feature_prep, type: list[str].
        label_cols: all column names must in the dataframe after label_prep, type: list[str].
        window:
        step:
        skip:
        etf_prep: callback function for etf dataset preprocessing before window rolling, type: function.
        feature_prep: callback function for feature preprocessing before window rolling, type: function.
        label_prep: callback function for label preprocessing before window rolling, type: function.
        feature_trans: callback function for feature transformation after window rolling, type: function.
        label_trans: callback function for label transformation after window rolling, type: function.
        read_csv_kwargs: optional kwargs for pd.read_csv api, type: dict.

    Return:
        Generator of tuple: (X, y)
    """
    etf = etf_prep(pd.read_csv(fpath, **read_csv_kwargs))

    x_skip = skip
    y_skip = window + x_skip - 1

    if feature_prep is None:
        feature_prep = (lambda df: df)
    features = feature_prep(etf)[feature_cols]
    if label_prep is None:
        label_prep = (lambda df: df)
    labels = label_prep(etf)[label_cols]

    x_gen = frame_rolling(features[feature_cols], window, step=step, skip=x_skip)
    y_gen = frame_rolling(labels[label_cols], 1, step=step, skip=y_skip)
    name_gen = naming(frame_rolling(etf, 1, step=step, skip=y_skip))
    for x, y, nm in zip(x_gen, y_gen, name_gen):
        yield (feature_trans(x, y), label_trans(x, y), nm)


def prepare_named_etf(fpath, *args, **kwargs):
    """TODO.

    Params:
        fpath: file path of the etf spreadsheet.
        feature_cols: all column names must in the dataframe after feature_prep, type: list[str].
        label_cols: all column names must in the dataframe after label_prep, type: list[str].
        window:
        step:
        skip:
        name_prep: callback function for giving name for each example.
        feature_prep: callback function for feature preprocessing before window rolling, type: function.
        label_prep: callback function for label preprocessing before window rolling, type: function.
        feature_trans: callback function for feature transformation after window rolling, type: function.
        label_trans: callback function for label transformation after window rolling, type: function.
        read_csv_kwargs: optional kwargs for pd.read_csv api, type: dict.
    """

    all_sample_gen = list(prepare_named_etf_gen(fpath, *args, **kwargs))
    X = [tp[0] for tp in all_sample_gen]  # noqa
    y = [tp[1] for tp in all_sample_gen]
    names = [tp[2] for tp in all_sample_gen]
    return X, y, names
