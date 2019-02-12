import numpy as np
import pandas as pd
from datetime import datetime


def _rsv(price, high=None, low=None, n_days=9):
    """Calculate the Raw Stochastic Value.
    TODO:
        1. check input object types (should be pd.Series)
        2. check if input shapes are matched. (len(price) == len(high) == len(low))

    Params:
        price: type: pd.Series
        high: type: pd.Series
        low: type: pd.Series 
        n_days: day period for calculation, type: int

    Return: feature: rsv, type: pd.Series
    """
    rolling_max = high.rolling(n_days, min_periods=0).max()
    rolling_min = low.rolling(n_days, min_periods=0).min()
    y = ((price - rolling_min) / (rolling_max - rolling_min)).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    y.name = 'rsv'
    return y  # alternative: pd.DataFrame(y, columns=['rsv'])
    

def _smoothing(seq, w, initial, name=None):
    """Smoothing calculation, such as k-index, d-index and ema.
    """
    y = [initial] * len(seq)
    wb = 1 - w
    for i, v in enumerate(seq[1:], 1):
        y[i] = w * v + wb * y[i - 1]
    out = pd.Series(y)
    if name:
        out.name = name
    return out


def _k(rsv, w=0.33, initial=0.5, name='k'):
    """Calculate K index.
    Params:
        rsv: rsv-index, type: pd.Series
        w: weighting, default: 0.33, type: float
        initial: initial k value, type: float

    Return: feature: `k`, type: pd.Series
    """
    return _smoothing(rsv, w, initial, name=name)


def _d(k, w=0.33, initial=0.5, name='d'):
    """Calculate D index.
    Params:
        k: k-index, type: pd.Series
        w: weighting, default: 0.33, type: float
        initial: initial k value, type: float

    Return: feature: `d`, type: pd.Series
    """
    return _smoothing(k, w, initial, name=name)


def ema(price, n_days=10, initial=None, name=None):
    if not initial:
        initial = price[0]
    if not name:
        name = f'ema{n_days}'
    w = 2 / float(n_days + 1)

    return _smoothing(price, w, initial, name=name)


def macd(price, short_term_days=12, long_term_days=26, dif_days=9):
    """
    TODO:
        1. should `ema_short`, `ema_long` be returned?
    Params:
        price: close price sequence in typical.
        short_term_days: default: 12, type: int
        long_term_days: default: 26, type: int
        dif_days: default: 9, type: int
    Return: feature: `dif`, `dem(macd)`, `osc`, type: pd.DataFrame
    """
    ema_short = ema(price, n_days=short_term_days)
    ema_long = ema(price, n_days=long_term_days)
    dif = ema_short - ema_long
    dem = ema(dif, n_days=dif_days, name='macd')
    osc = dif - dem

    dif.name = 'dif'
    osc.name = 'osc'
    return pd.concat([dif, dem, osc], axis=1)


def kd_rsv(price, high=None, low=None, n_days=9, k_init=0.5, k_w=0.33, d_init=0.5, d_w=0.33):
    """
    Params:
        price: stock price sequence, type: pd.Series
        high: high price sequence, type: pd.Series
        low: low price sequece, type: pd.Series
        n_days: day period for analysis, default: 9, type: int
        k_init: initial value for k-index sequence, default: 0.5,  type: float
        k_w: momentum weighting for generating new k, default: 0.33, type: float
        d_init: initial value for d-index sequence, default: 0.5,  type: float
        d_w: momentum weighting for generating new d, default: 0.33, type: float
    Return: features: rsv, k, d, type: pd.DataFrame
    """
    rsv = _rsv(price, high=high, low=low, n_days=n_days)
    k = _k(rsv, w=k_w, initial=k_init)
    d = _d(k, w=d_w, initial=d_init)

    return pd.concat([rsv, k, d], axis=1)


def _diff(seq):
    """
    Params:
       seq, type: pd.Series
    Return: type: pd.Series
    """
    return seq.diff().fillna(0)


def rsi(price, n_days=7):
    """
    TODO: is `diff` necessary to return?
    Params:
        price: stock price sequence, type: pd.Series
        n_days: day period for analysis, default: 7, type: int
    Return: features: upward, downward, rs and rsi, type: pd.DataFrame
    """
    diff = _diff(price)
    # diff.name = 'diff'
    upward = diff.rolling(n_days, min_periods=0).apply(lambda frame: frame.dot(frame > 0) / n_days)
    upward.name = 'upward'
    downward = diff.rolling(n_days, min_periods=0).apply(lambda frame: -frame.dot(frame < 0) / n_days)
    downward.name = 'downward'
    rs = (upward / (upward + downward)).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    rs.name = 'rs'
    rsi = rs / (1.0 + rs)
    rsi.name = 'rsi'

    return pd.concat([upward, downward, rs, rsi], axis=1)


def _candle(open_price=None, close_price=None, high=None, low=None):
    """Calculate candle stick features.
    TODO:
        1. Dimension check.
    Params:
        *all, type: pd.Series
    Return:
        kbody, up_shadow, low_shadow
    """
    prices = pd.concat([close_price, open_price], axis=1)
    kbody = close_price - open_price
    kbody_top = prices.max(axis=1)
    kbody_bottom = prices.min(axis=1)
    up_shadow = high - kbody_top
    low_shadow = kbody_bottom - low

    kbody.name = 'kbody'
    kbody_top.name = 'kbody_top'
    kbody_bottom.name = 'kbody_bottom'
    up_shadow.name = 'up_shadow'
    low_shadow.name = 'low_shadow'

    return pd.concat([kbody, kbody_top, kbody_bottom, up_shadow, low_shadow], axis=1)


def _open_gap(open_price=None, close_price=None):
    """
    Return: gap, type: pd.Series
    """
    prev_close = close_price.shift(periods=1).fillna(open_price[0])

    gap = open_price - prev_close
    gap.name = 'open_gap'

    return gap


def candle_stick(open_price=None, close_price=None, high=None, low=None):
    """Calculate candle stick features.
    TODO:
        1. Dimension check.
    Params:
        *all, type: pd.Series
    Return:
        kbody, up_shadow, low_shadow
    """
    candles = _candle(open_price=open_price, close_price=close_price, high=high, low=low)
    gap = _open_gap(open_price=open_price, close_price=close_price)

    return pd.concat([candles, gap], axis=1)


def true_range(price, high=None, low=None):
    prev_price = price.shift(1).fillna(price[0])
    tr = pd.concat([high - low, (high - prev_price).abs(), (low - prev_price).abs()], axis=1).max(axis=1)
    tr.name = 'tr'

    return tr


def trma(price, high=None, low=None, n_days=20):
    tr = true_range(price, high=high, low=low)
    trma_ = tr.rolling(n_days, min_periods=0).mean()
    trma_.name = 'trma'

    return trma_


def atr(price, high=None, low=None, n_days=20, w=0.33):
    trma_ = trma(price, high=high, low=low, n_days=n_days)
    return _smoothing(trma_, w, trma_[0], name='atr')


def atr_std(price, high=None, low=None, n_days=20, w=0.33, fillna=0, atr_w=1.5, std_w=2.0):
    std = price.rolling(n_days).std().fillna(fillna)
    std.name = f'price_std_{n_days}'

    atr_ = atr(price, high=high, low=low, n_days=n_days, w=w)
    atr_.name = f'atr_{n_days}'

    atr_std_ = atr_w * atr_ - std_w * std
    atr_std_.name = f'atr_std_{n_days}'

    return pd.concat([std, atr_, atr_std_], axis=1)


def vr_obv(price, volume, n_days=12, vol_w=0.5):
    dif =  price.diff().fillna(0)
    vol_upward = ((1.0 * (dif >= 0) + vol_w) * volume).rolling(n_days, min_periods=0).sum()
    vol_downward = ((1.0 * (dif < 0) + vol_w) * volume).rolling(n_days, min_periods=0).sum()

    vr = (vol_upward / vol_downward).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    sign = np.where(dif >= 0, 1, -1)

    obv = (sign * volume).cumsum()

    obv_ma = obv.rolling(n_days, min_periods=0).mean()
    obv_ma_diff = obv - obv_ma

    vr.name = 'vr'
    obv.name = 'obv'
    obv_ma.name = f'obv_ma{n_days}'
    obv_ma_diff.name = f'obv_ma{n_days}_diff'

    return pd.concat([vr, obv, obv_ma, obv_ma_diff], axis=1)


def trend(price, short_term_days=5, long_term_days=20):
    trend_up = 1 * (price.rolling(short_term_days, 0).max() - price.rolling(long_term_days, 0).max().shift(short_term_days) > 0)
    trend_down = 1 * (price.rolling(short_term_days, 0).min() - price.rolling(long_term_days, 0).min().shift(short_term_days) < 0)
    trend_up.name = 'trend_up'
    trend_down.name = 'trend_down'

    return pd.concat([trend_up, trend_down], axis=1)
