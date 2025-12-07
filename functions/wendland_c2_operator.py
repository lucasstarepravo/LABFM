import numpy as np
from tqdm import tqdm
from functions.nodes import neighbour_nodes_kdtree
from scipy.spatial import cKDTree
import math
from functions.plot import plot_kernel

def wendland_c2_sph(neighbours_r, h):
    q = neighbours_r/h

    if (q < 0).any() or (q > 2).any():
        raise ValueError("r must be >0 and <=2")

    w_ji = (7/(math.pi*h**2) ) * (1 - q)**4 * (1 + 4 * q)
    return w_ji

def wendland_c2_deriv_rad(neighbours_r, h):
    q = neighbours_r/h

    if (q < 0).any() or (q > 2).any():
        raise ValueError("r must be >0 and <=2")

    c = (-140 * neighbours_r)/(math.pi * h**4)
    w_ji = c * (1 - q) ** 3
    return w_ji

def wendland_c2_deriv(neighbours_r, neigh_xy_d, h, deriv):
    if deriv.lower() not in ['dx', 'dy']:
        raise ValueError("deriv must be either 'dx' or 'dy'")

    q = neighbours_r / h
    if (q < 0).any() or (q > 2).any():
        raise ValueError("r must be >0 and <=2")

    deriv = deriv.lower()
    if deriv == 'dx':
        dist = neigh_xy_d[:, 0]
    else:
        dist = neigh_xy_d[:, 1]

    c = (-140) / (np.pi * h ** 3)
    w_ji = c * q * (1 - q) ** 3 * dist
    w_ji[1:] /=  neighbours_r[1:]

    return w_ji


def wendland_c2_laplacian(neighbours_r, h):
    q = neighbours_r/h
    if (q < 0).any() or (q > 2).any():
        raise ValueError("r must be >0 and <=2")

    c = (-140/(math.pi * h ** 4))

    w_ji = c * (1 - q) ** 2 * (2 - 5 * q)
    return w_ji


def wendlandc2_weights(coordinates, h, total_nodes):
    tree = cKDTree(coordinates)

    neigh_r_dict    = {}
    neigh_coor_dict = {}
    neigh_xy_dist   = {}
    density_dict    = {}
    weights_x       = {}
    weights_y       = {}
    #weights_laplace = {}
    #weights_r       = {}

    support_radius = 2 * h
    #plot_xy = []


    for ref_x, ref_y in tqdm(coordinates, desc="Calculating Wendland Weights for " + str(total_nodes) + ", ", ncols=100):
        # int his current form the density of all node swill be computed, but we only need up to the neighbours of the edge nodes
        ref_node = (ref_x, ref_y)
        (neigh_r_d,
         neigh_xy_d,
         neigh_coor_dict[ref_node]) = neighbour_nodes_kdtree(coordinates,
                                                             ref_node,
                                                             support_radius,
                                                             tree,
                                                             max_neighbors=100)
        density_dict[ref_node] = np.ones(shape=neigh_r_d.shape) @ wendland_c2_sph(neigh_r_d, support_radius)

        if ref_x > 0.5 or ref_x < -0.5 or ref_y > 0.5 or ref_y < -0.5: continue

        #plot_xy.append(neigh_xy_d)
        neigh_xy_dist[ref_node] = neigh_xy_d
        neigh_r_dict[ref_node] = neigh_r_d
        weights_x[ref_node] = wendland_c2_deriv(neigh_r_d, neigh_xy_d, support_radius, 'dx')
        weights_y[ref_node] = wendland_c2_deriv(neigh_r_d, neigh_xy_d, support_radius, 'dy')
        #weights_laplace[ref_node] = wendland_c2_laplacian(neigh_r_d, support_radius)
        #weights_r[ref_node] = wendland_c2_deriv_rad(neigh_r_d, support_radius)

    #plot_xy_np = np.array(plot_xy)
    #plot_xy_np /= support_radius
    #plot_kernel(plot_xy_np)

    return weights_x, weights_y, density_dict, neigh_coor_dict, neigh_r_dict, neigh_xy_dist#, weights_laplace, weights_r