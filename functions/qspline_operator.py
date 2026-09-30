import numpy as np
from tqdm import tqdm
from functions.nodes import neighbour_nodes_kdtree
from scipy.spatial import cKDTree
import time

DTYPE = np.float32

def quintic_spline(neighbours_r, h):
    norm = neighbours_r / h
    w = []

    for s in norm:
        if 0 <= s < 1:
            ww = (3 - s) ** 5 - 6 * (2 - s) ** 5 + 15 * (1 - s) ** 5
        elif 1 <= s < 2:
            ww = (3 - s) ** 5 - 6 * (2 - s) ** 5
        elif 2 <= s < 3:
            ww = (3 - s) ** 5
        else:
            ww = 0.0
        w.append(ww)

    w = (7/(478*np.pi*h**2))*np.array(w)

    return w

def quintic_spline_deriv(neighbours_r, neigh_xy_d, h, s, deriv):
    q = neighbours_r / h

    if deriv == 'dx':
        comp = neigh_xy_d[:, 0]
    elif deriv == 'dy':
        comp = neigh_xy_d[:, 1]
    else:
        raise ValueError('deriv must be dx or dy')

    xy_div_r = np.zeros_like(neighbours_r)
    mask = neighbours_r > 0
    xy_div_r[mask] = comp[mask] / neighbours_r[mask]

    ww = np.zeros_like(q)

    m0 = (q >= 0) & (q < 1)
    m1 = (q >= 1) & (q < 2)
    m2 = (q >= 2) & (q < 3)

    t3 = (3 - q)
    t2 = (2 - q)
    t1 = (1 - q)

    ww[m2] = -5 * t3[m2]**4
    ww[m1] = -5 * t3[m1]**4 + 30 * t2[m1]**4
    ww[m0] = -5 * t3[m0]**4 + 30 * t2[m0]**4 - 75 * t1[m0]**4

    ww = (xy_div_r / h) * ww
    return (7 / (478 * np.pi * h**2)) * ww * s**2


def qspline_weights(coordinates, h, total_nodes, s):
    tree = cKDTree(coordinates)

    neigh_r_dict    = {}
    neigh_coor_dict = {}
    neigh_xy_dist   = {}
    weights_x       = {}
    weights_y       = {}
    weights_laplace = {}
    support_radius = 3 * h

    for ref_x, ref_y in tqdm(coordinates, desc="Calculating Quintinc Spline Weights for " + str(total_nodes) + ", ", ncols=100):
        if ref_x > 0.5 or ref_x < -0.5 or ref_y > 0.5 or ref_y < -0.5: continue
        ref_node = (ref_x, ref_y)
        (neigh_r_d,
         neigh_xy_d,
         neigh_coor_dict[ref_node]) = neighbour_nodes_kdtree(coordinates,
                                                             ref_node,
                                                             support_radius,
                                                             tree,
                                                             max_neighbors=100)

        neigh_r_d = np.asarray(neigh_r_d, dtype=DTYPE)
        neigh_xy_d = np.asarray(neigh_xy_d, dtype=DTYPE)
        neigh_xy_dist[ref_node] = neigh_xy_d
        neigh_r_dict[ref_node] = neigh_r_d
        weights_x[ref_node] = -quintic_spline_deriv(neigh_r_d, neigh_xy_d, h, s, 'dx')
        weights_y[ref_node] = -quintic_spline_deriv(neigh_r_d, neigh_xy_d, h, s, 'dy')
        weights_laplace[ref_node] =   (2 * (neigh_xy_d[1:, 0] * weights_x[ref_node][1:] +
                                            neigh_xy_d[1:, 1] * weights_y[ref_node][1:]) /  neigh_r_d[1:] ** 2)
        weights_laplace[ref_node] = np.concatenate((np.zeros(1, dtype=DTYPE), weights_laplace[ref_node]))

    return weights_x, weights_y, weights_laplace, neigh_coor_dict, neigh_xy_dist
