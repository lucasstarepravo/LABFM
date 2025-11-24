import numpy as np
from tqdm import tqdm
from functions.nodes import neighbour_nodes_kdtree
from scipy.spatial import cKDTree
import math

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

def quintic_spline_deriv(neighbours_r, neigh_xy_d, h, deriv):
    s_array = neighbours_r / h
    w = []

    if deriv == 'dx':
        xy_div_r = np.zeros_like(neighbours_r)
        mask = neighbours_r > 0
        xy_div_r[mask] = neigh_xy_d[mask, 0] / neighbours_r[mask]
    elif deriv == 'dy':
        xy_div_r = np.zeros_like(neighbours_r)
        mask = neighbours_r > 0
        xy_div_r[mask] = neigh_xy_d[mask, 1] / neighbours_r[mask]
    else:
        raise ValueError('deriv must be dx or dy')

    for i in range(s_array.shape[0]):
        s = s_array[i]
        if 0 <= s < 1:
            ww = -5*(3-s)**4 + 30*(2-s)**4 - 75*(1-s)**4
        elif 1 <= s < 2:
            ww = -5*(3-s)**4 + 30*(2-s)**4
        elif 2 <= s < 3:
            ww = -5*(3-s)**4
        else:
            ww = 0

        ww = (xy_div_r[i] / h) * ww
        w.append(ww)

    w = (7/(478*np.pi*h**2))*np.array(w)
    return w


def quintic_spline_laplace(neighbours_r, h):
    norm = neighbours_r / h
    w = []

    for i in range(norm.shape[0]):
        s = norm[i]
        if 0 <= s < 1:
            Fpp = 20*(3 - s)**3 - 120*(2 - s)**3 + 300*(1 - s)**3
            Fp  = -5*(3 - s)**4 + 30*(2 - s)**4 - 75*(1 - s)**4
        elif 1 <= s < 2:
            Fpp = 20*(3 - s)**3 - 120*(2 - s)**3
            Fp  = -5*(3 - s)**4 + 30*(2 - s)**4
        elif 2 <= s < 3:
            Fpp = 20*(3 - s)**3
            Fp  = -5*(3 - s)**4
        else:
            Fpp = 0.0
            Fp  = 0.0

        ww = (Fpp + (Fp / s if s > 0 else 0.0)) / (h**2)
        w.append(ww)

    w = (7 / (478 * np.pi * h ** 2)) * np.array(w)
    return w

def qspline_weights(coordinates, h, total_nodes):
    tree = cKDTree(coordinates)

    neigh_r_dict    = {}
    neigh_coor_dict = {}
    density_dict    = {}
    weights_x       = {}
    weights_y       = {}
    weights_laplace = {}
    support_radius = 2.2 * h




    for ref_x, ref_y in tqdm(coordinates, desc="Calculating Quintinc Spline Weights for " + str(total_nodes) + ", ", ncols=100):
        # int his current form the density of all node swill be computed, but we only need up to the neighbours of the edge nodes
        ref_node = (ref_x, ref_y)
        (neigh_r_d,
         neigh_xy_d,
         neigh_coor_dict[ref_node]) = neighbour_nodes_kdtree(coordinates,
                                                             ref_node,
                                                             support_radius,
                                                             tree,
                                                             max_neighbors=200)
        density_dict[ref_node] = np.ones(shape=neigh_r_d.shape) @ quintic_spline(neigh_r_d, h)
        if ref_x > 0.5 or ref_x < -0.5 or ref_y > 0.5 or ref_y < -0.5: continue

        neigh_r_dict[ref_node] = neigh_r_d
        weights_x[ref_node] = quintic_spline_deriv(neigh_r_d, neigh_xy_d, h, 'dx')
        weights_y[ref_node] = quintic_spline_deriv(neigh_r_d, neigh_xy_d, h, 'dy')
        weights_laplace[ref_node] = quintic_spline_laplace(neigh_r_d, h)


    return weights_x, weights_y, weights_laplace, density_dict, neigh_coor_dict, neigh_r_dict
