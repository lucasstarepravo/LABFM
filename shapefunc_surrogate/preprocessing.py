from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np


def non_dimension(features, labels, dx, dtype='laplace'):
    '''
    This function uses the stencil size which is 1.5dx to normalize the feature vector
    :param features:
    :param labels:
    :param dx:
    :param dtype:
    :return:
    '''
    if dtype not in ['laplace', 'x', 'y']:
        raise ValueError('dtype variable must be "laplace", "x" or "y"')
    dx = 1.5 * dx  # The 1.5 is obtained from the Fortran code, which is the ratio between h/s
    if dtype == 'laplace':
        h_scale_w = dx ** 2
        h_scale_xy = dx
    else:
        h_scale_w = dx
        h_scale_xy = dx

    f_mean = np.mean(features, axis=1, keepdims=True)
    stand_feature = (features - f_mean) / h_scale_xy

    l_mean = np.mean(labels, axis=1, keepdims=True)
    stand_label = (labels - l_mean) * h_scale_w
    return stand_feature, stand_label, f_mean, l_mean, h_scale_xy, h_scale_w