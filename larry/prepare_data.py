import numpy as np
import pandas as pd

from numpy.lib.stride_tricks import as_strided as strided

def get_sliding_window(df, n_days):
    '''
    df: feature data set
    length, features: shape of df,
    lenth: # of rows,
    features: # of features
    from df (shape: (m,n)) to make training data with shape (m-n_days+1, n_days, n)


    '''
    a = df.values
    print ('a.strides', a.strides)
    s0,s1 = a.strides #(bytes needed to stride)
    print ('a.strides', a.strides)
    print ('\n')
    print ('s0', s0, 's1', s1)
    print ('\n')
    length,features = a.shape
    print ('length',length,'features',features)
    print ('\n')
    out = strided(a,shape=(length-n_days+1,n_days,features),strides=(s0,s0,s1))

    return out

    
