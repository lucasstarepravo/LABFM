import numpy as np
import math

def rescale_h(actual_l, pred_l, feat_subset, f_mean, l_mean, h_scale_xy, h_scale_w, test_index):
    tst_mean_l = l_mean[test_index]

    sc_actual_l = actual_l / h_scale_w + tst_mean_l
    sc_pred_l = pred_l / h_scale_w + tst_mean_l

    tst_mean_f = f_mean[test_index]

    sc_feat = feat_subset.reshape(feat_subset.shape[0], -1, 2) * h_scale_xy + tst_mean_f

    return sc_actual_l, sc_pred_l, sc_feat


def d_2_c(coor, test_index, scaled_feat):
    '''
    The ANN features are the x and y distances of the neighbour nodes to the reference node. This function takes the
    whole coordinates original vector, and the test_index vector obtained from the tran_test_split and, finds the
    coordinates of the test nodes
    :param coor:
    :param test_index:
    :param scaled_feat:
    :return:
    '''
    zeros = np.zeros((int(scaled_feat.shape[0] * 2))).reshape(scaled_feat.shape[0], -1, 2)
    scaled_feat = np.concatenate((zeros, scaled_feat), axis=1)
    tst_coor = coor[test_index, :]
    tst_coor = tst_coor.reshape(tst_coor.shape[0], -1, 2)
    d_2_c = scaled_feat + tst_coor
    return d_2_c