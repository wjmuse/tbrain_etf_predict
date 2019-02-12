import os


src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw/groupbycode/trainingset/'))
this_dir = os.path.dirname(os.path.abspath(__file__))


# Selected features
cont_feature_cols = ['close',]
cate_feature_cols = []

# Selected targets
label_cols = [f'y_{i+1}' for i in range(5)]


window = 6
n_experiment = 1

