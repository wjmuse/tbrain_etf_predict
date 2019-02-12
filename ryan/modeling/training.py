import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import datetime
import argparse
from etf_tools.parsers import training_parser

PJ = os.path.join
PABS = os.path.abspath



# training_parser.add_argument('--data-dir', nargs='?', type=str, ...) 
# training_parser.add_argument('--data-loader', nargs='?', type=str, ...)
# training_parser.add_argument('--model-factory', nargs='?', type=str, ...)


data_dir =  PABS(PJ(os.path.dirname(__file__), '../../data/raw/groupbycode/all/'))
col_dtypes = OrderedDict(code=str, date=str, name=str, open=float, high=float, low=float, close=float, volume=int, weekday=int)




if __name__ == '__main__':
    args = training_parser.parse_args()
    code = args.code
    date = args.datemark
    sample_file = PJ(data_dir, f'{code}.csv')
    etf_df = pd.read_csv(sample_file, names=col_dtypes.keys(), dtype=col_dtypes, skiprows=1)

    targetrow = etf_df[etf_df.date == date]
    print(targetrow)
