import numpy as np


def test_function(coordinates):
    """
    :param nodes:
    :return:
    """

    x = coordinates[:, 0] - .1453
    y = coordinates[:, 1] - .16401
    phi = 1 + (x * y) ** 4 + (x * y) ** 8 + (x + y) + (x ** 2 + y ** 2) + (x ** 3 + y ** 3) + (x ** 4 + y ** 4) + \
          (x ** 5 + y ** 5) + (x ** 6 + y ** 6)
    phi_dict = {(coordinates[i, 0], coordinates[i, 1]): phi[i] for i in range(phi.shape[0])}
    return phi_dict


def dif_analytical(coordinates, derivative):
    """
    :param coordinates:
    :param derivative:
    :return:
    """
    if derivative not in ['dtdx', 'dtdy']:
        raise ValueError("Invalid derivative type")

    # Determine the variable of interest based on the derivative
    if derivative == 'dtdx':
        var = coordinates[:, 0] - .1453
        const = coordinates[:, 1] - .16401
    else:
        var = coordinates[:, 1] - .16401
        const = coordinates[:, 0] - .1453


    # Calculate the terms using a loop
    result = 4 * var ** 3 * const ** 4 + 8 * var ** 7 * const ** 8 + 1 + 2 * var + 3 * var ** 2 + 4 * var ** 3 + 5 * var ** 4 + 6 * var ** 5
    result_dic = {(coordinates[i, 0], coordinates[i, 1]): result[i] for i in range(result.shape[0])}
    return result_dic


def laplace_phi(coordinates):
    """
    :return:
    """
    x = coordinates[:, 0] - .1453
    y = coordinates[:, 1] - .16401

    # Terms from the derived Laplacian of ϕ
    term1 = 4
    term2 = 6 * x
    term3 = 12 * x ** 2
    term4 = 20 * x ** 3
    term5 = 30 * x ** 4
    term6 = 6 * y
    term7 = 12 * y ** 2
    term8 = 12 * x ** 4 * y ** 2
    term9 = 20 * y ** 3
    term10 = 30 * y ** 4
    term11 = 12 * x ** 2 * y ** 4
    term12 = 56 * x ** 8 * y ** 6
    term13 = 56 * x ** 6 * y ** 8

    result = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13
    result_dic = {(coordinates[i, 0], coordinates[i, 1]): result[i] for i in range(result.shape[0])}
    return result_dic


def dif_do(weights, surface_value, derivative):
    """
    :param weights:
    :param surface_value:
    :param derivative:
    :return:
    """

    neigh = weights._neigh_coor

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
    for ref_node in neigh:
        surface_dif = np.array([surface_value[tuple(n_node)] - surface_value[tuple(ref_node)] for n_node in neigh[ref_node]]).reshape(1,-1)
        w_ref_node  = w_dif[ref_node]
        dif_approx[ref_node] = np.dot(surface_dif, w_ref_node)

    return dif_approx


def dif_gnn(weights, surface_value, derivative):
    """
    :param weights:
    :param surface_value:
    :param derivative:
    :return:
    """

    neigh = weights._neigh_coor

    if derivative.lower() not in ["laplace"]:
        raise ValueError("The valid_string argument must be 'laplace' ")

    if derivative == 'Laplace':
        w_dif = weights.laplace

    # This calculates the approximation of the derivative
    dif_approx = {}
    for ref_node in neigh:
        surface_dif = np.array([surface_value[tuple(n_node)] - surface_value[tuple(ref_node)] for n_node in neigh[ref_node]]).reshape(1,-1)
        try:
            w_ref_node  = w_dif[ref_node]
        except IndexError:
            h = ref_node
            print('hi')
        dif_approx[ref_node] = np.dot(surface_dif, w_ref_node)

    return dif_approx

def deriv_sph(weights, surface_value, derivative):
    if derivative not in ["dtdx", "dtdy"]:
        raise ValueError("The valid_string argument must be 'dtdx', or 'dtdy'")

    if derivative == "dtdx":
        w_dif = weights.x
    else:
        w_dif = weights.y

    neigh = weights._neigh_coor
    rho = weights.rho
    dif_approx = {}

    for ref_node in neigh:
        x = ref_node[0]
        y = ref_node[1]
        if x > 1 or x < 0 or y > 1 or y < 0:
            continue
        surface_diff = []
        for nn in neigh[ref_node]:
            surface_diff.append(surface_value[tuple(nn)] - surface_value[ref_node])
        surface_diff = np.array(surface_diff)
        dif_approx[ref_node] = -(1 / rho[ref_node]) * (np.dot(surface_diff, w_dif[ref_node]))

    return dif_approx


def lap_sph(sph_weights_attr, surface_value):
    neigh_coor = sph_weights_attr._neigh_coor
    neigh_r    = sph_weights_attr._neigh_r
    lap_w      = sph_weights_attr.laplace
    rho        = sph_weights_attr.rho

    dif_approx = {}

    for ref_node in neigh_coor:
        x = ref_node[0]
        y = ref_node[1]
        if x > 1 or x < 0 or y > 1 or y < 0:
            continue
        temp_ls = []
        for i in range(neigh_coor[ref_node].shape[0]):
            nn = neigh_coor[ref_node][i]
            delta_x = neigh_r[ref_node][i] / neigh_r[ref_node][i] ** 2 if neigh_r[ref_node][i] != 0 else 0

            surface_diff = (surface_value[tuple(nn)] - surface_value[ref_node]) * delta_x
            surface_diff = surface_diff / rho[tuple(nn)] # this uses density of all nodes, not just the centrla ones
            temp_ls.append(surface_diff)

        temp_ls = np.array(temp_ls)
        dif_approx[ref_node] = np.dot(temp_ls, lap_w[ref_node])

    return dif_approx
