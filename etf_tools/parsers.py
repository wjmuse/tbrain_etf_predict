import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import datetime
import argparse



etf_parser = argparse.ArgumentParser(prog='etf')
etf_parser.add_argument('--n-experiment', nargs='?', type=int, default=1, help='How many experiments to take. Each experiment runs a single training process independently.')
etf_parser.add_argument('--code', nargs='?', type=str, default='0050', help='ETF code number.')
etf_parser.add_argument('--datemark', nargs='?', type=str, default='20180504', help='Use the data on this date as test set.')
etf_parser.add_argument('--keep-model', nargs='?', type=bool, default=True, help='Whether to keep the trained model from each expirement')
etf_parser.add_argument('--overwrite', nargs='?', type=bool, default=False, help='Whether to overwrite the existing files/data.')
