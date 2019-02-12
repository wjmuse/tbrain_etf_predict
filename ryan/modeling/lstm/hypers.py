import os


src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw/groupbycode/trainingset/'))
this_dir = os.path.dirname(os.path.abspath(__file__))


# Selected features
cont_feature_cols = [
    'open', 'high', 'low', 'close', 'volume',
    'ema12', 'ema26', 'rsv', 'k', 'd', 'upward',
    'downward', 'rsi', 'open_gap', 'macd', 'osc', 'kbody',
    'price_std_20', 'atr_20', 'atr_std_20',
    'vr', 'obv', 'obv_ma12', 'obv_ma12_diff'
    # 'trend_cumsum'
]
cate_feature_cols = ([f'month_{i + 1}' for i in range(12)]
                      + [f'week_{i + 1}' for i in range(5)]
		      + ['trend_up', 'trend_down'])


# Selected targets
label_cols = [f'y_{i+1}' for i in range(5)]


# Hyperparams
feature_dim = len(cont_feature_cols + cate_feature_cols)
num_classes = 1
n_lastdays = 5
output_step = len(label_cols)
timestep = window = 20
batchsize = 128
epochs = 200
dropout = 0.0
n_experiment = 5
ckpt_step = 10
ckpt_skip = 10
