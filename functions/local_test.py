import numpy as np
import random


def generate_random_numbers(seed, length):
    random.seed(seed)
    return np.array([round(random.uniform(0, 0.5), 3) for _ in range(length)])


def test_function2(coordinates, weights, derivative='dtdx'):
    """
    :param nodes:
    :return:
    phi: is the surface field
    """
    if derivative not in ['dtdx', 'dtdy', 'Laplace']:
        raise ValueError("Invalid derivative type")

    if derivative == 'dtdx':
        shift_x = generate_random_numbers(1, coordinates.shape[0])
        shift_dict = {(coordinates[i, 0], coordinates[i, 1]): shift_x[i] for i in range(len(shift_x))}
        x = coordinates[:, 0] - shift_x
        phi = x
    elif derivative == 'dtdy':
        shift_y = generate_random_numbers(2, coordinates.shape[0])
        shift_dict = {(coordinates[i, 0], coordinates[i, 1]): shift_y[i] for i in range(len(shift_y))}
        y = coordinates[:, 1] - shift_y
        phi = y
    elif derivative == 'Laplace':   # If it's laplace the function need to be quadratic
        shift_x = generate_random_numbers(1, coordinates.shape[0])
        shift_y = generate_random_numbers(2, coordinates.shape[0])
        shift_dict = {(coordinates[i, 0], coordinates[i, 1]): [shift_x[i], shift_y[i]] for i in range(len(shift_x))}
        x = coordinates[:, 0] - shift_x
        y = coordinates[:, 1] - shift_y
        phi = x**2 + y**2

    # Calculating the surface value of the neighbours of each reference node.
    # I want neigh_surface to be accessed as neigh_surface[(ref_node)][(neighbour_node)]

    neigh_coor = weights._neigh_coor
    neigh_surface = {}

    if derivative == 'dtdx':
        for ref_node in neigh_coor:
            temp_dict = {}
            for n_node in neigh_coor[ref_node]:
                temp_dict[tuple(n_node)] = n_node[0] - shift_dict[ref_node]
            neigh_surface[ref_node] = temp_dict
    elif derivative == 'dtdy':
        for ref_node in neigh_coor:
            temp_dict = {}
            for n_node in neigh_coor[ref_node]:
                temp_dict[tuple(n_node)] = n_node[1] - shift_dict[ref_node]
            neigh_surface[ref_node] = temp_dict
    elif derivative == 'Laplace':
        for ref_node in neigh_coor:
            temp_dict = {}
            for n_node in neigh_coor[ref_node]:
                temp_dict[tuple(n_node)] = (n_node[0] - shift_dict[ref_node][0])**2 + (n_node[1] - shift_dict[ref_node][1])**2
            neigh_surface[ref_node] = temp_dict


    phi_dict = {(coordinates[i, 0], coordinates[i, 1]): phi[i] for i in range(phi.shape[0])}
    return phi_dict, neigh_surface


def dif_analytical2(coordinates, derivative):
    """
    :param coordinates:
    :param derivative:
    :return:
    """
    if derivative not in ['dtdx', 'dtdy','Laplace']:
        raise ValueError("Invalid derivative type")

    if derivative == 'dtdx' or derivative == 'dtdy':
        result = np.ones(len(coordinates))
    elif derivative == 'Laplace':
        result = 4 * np.ones(len(coordinates))

    # Calculate the terms using a loop
    result_dic = {(coordinates[i, 0], coordinates[i, 1]): result[i] for i in range(result.shape[0])}
    return result_dic


def dif_do2(weights, surface_value, neigh_surface, derivative):
    """
    :param surface_value:
    :param derivative:
    :return:
    """

    neigh_coor = weights._neigh_coor

    if derivative not in ["dtdx", "dtdy", "Laplace"]:
        raise ValueError("The valid_string argument must be 'dtdx', 'dtdy' or 'Laplace' ")

    if derivative == "dtdx":
        w_dif = weights.x
    elif derivative == "dtdy":
        w_dif = weights.y
    elif derivative == 'Laplace':
        w_dif = weights.laplace
    # This calculates the approximation of the derivative
    dif_approx = {}

    for ref_node in neigh_coor:
        surf_dif = []
        for n_node in neigh_coor[ref_node]:
            surf_dif.append(neigh_surface[ref_node][tuple(n_node)] - surface_value[ref_node])
        w_ref_node = w_dif[ref_node].reshape(-1, 1)
        dif_approx[ref_node] = np.dot(np.array(surf_dif).reshape(1,-1), w_ref_node)

    return dif_approx
