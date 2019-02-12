import pandas as pd
import numpy as np
import math
from sklearn.utils import shuffle as sk_shuffle


class EstimatorBucket:

    def draw(self):
        """
        Return 
            estimator/model, type: sklearn.base.XXXMixin
        """
        raise NotImplementedError


def evaluate_stacking(est_bucket, stacking_strategy,
                      x_val=None, y_val=None, metrics=[], n_stack=3,
                      bucket_kwargs={}, strategy_kwargs={}):
    """
    Params:
        est_bucket: draw estimator from, type: EstimatorBucket
        stacking_strategy: type: function
        x_val: validation set to be evaluated, type: array-like
        y_val: validation set to be evaluated, type: array-lile
        metrics: type: list of function
        n_stack: type: int
    return pd.Dataframe
    """
    if not (isinstance(metrics, list) or isinstance(metrics, tuple)):
        raise TypeError('Metrics should wrapped by list/tuple')
    elif len(metrics) == 0:
        ValueError('Must provide at least one metric function.')

    if x_val is None or y_val is None:
        raise ValueError('Must provide the validation set: x_val/y_val')

    est = []
    stack_results = []
    for i in range(n_stack):
        est.append(est_bucket.draw(**bucket_kwargs))

        weighting = stacking_strategy(est, **strategy_kwargs)
        w_sum = sum(weighting)
        scores =  {f'{met.__name__}': sum([w * met(y_val, e.predict(x_val)) for w, e in zip(weighting, est)]) / w_sum for met in metrics}

        stack_results.append(scores)
    out = pd.DataFrame(stack_results).T
    out.columns = [f'stk_{i + 1}' for i in range(n_stack)]

    return out


def frame_rolling(df, window, step=1, skip=0, min_periods=None):
    """
    Params:
        df: dataframe to be rolling.
        window: rolling window size.
        step: step size between rolling.
        skip: skip size from head of dataframe
    Return:
        Generator of input type
    """
    min_periods = min_periods or window
    df_skipped = df[skip:]
    df_sliced = df_skipped[: window]
    i_step = 0
    while df_sliced.shape[0] >= min_periods:
        yield df_sliced
        i_step += step
        df_sliced = df_skipped[i_step : i_step + window]


def prepare_etf_gen(etf, feature_cols, label_cols, window=5, step=1, skip=0,
                    etf_prep=lambda df: df.dropna(),
                    feature_prep=lambda df: df, label_prep=lambda df: df,
                    feature_trans=lambda df: df, label_trans=lambda df: df,
                    read_csv_kwargs={}):
    """
    Pipeline sequence: read_csv --> etf_prep --> (feature_prep, label_prep) --> (select feature_cols, select label_cols) --> (feature_trans, label_trans)
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
    #etf = etf_prep(pd.read_csv(fpath, **read_csv_kwargs))

    x_skip = skip
    y_skip = window + x_skip - 1

    if feature_prep is None:
        feature_prep=lambda df: df
    features = feature_prep(etf)[feature_cols]
    if label_prep is None:
        feature_prep=lambda df: df
    labels = label_prep(etf)[label_cols]

    x_gen = frame_rolling(features[feature_cols], window, step=step, skip=x_skip)
    y_gen = frame_rolling(labels[label_cols], 1, step=step, skip=y_skip)

    for x, y in zip(x_gen, y_gen):
        yield (feature_trans(x), label_trans(y))


def prepare_etf(fpath, *args, **kwargs):
    """
    Params:
        fpath: file path of the etf spreadsheet.
        feature_cols: all column names must in the dataframe after feature_prep, type: list[str].
        label_cols: all column names must in the dataframe after label_prep, type: list[str].
        window:
        step:
        skip:
        feature_prep: callback function for feature preprocessing before window rolling, type: function.
        label_prep: callback function for label preprocessing before window rolling, type: function.
        feature_trans: callback function for feature transformation after window rolling, type: function.
        label_trans: callback function for label transformation after window rolling, type: function.
        read_csv_kwargs: optional kwargs for pd.read_csv api, type: dict.
    """
    all_sample_gen = list(prepare_etf_gen(fpath, *args, **kwargs))
    X = np.asarray([tp[0].values for tp in all_sample_gen])
    y = np.asarray([tp[1].values for tp in all_sample_gen])

    return X, y


