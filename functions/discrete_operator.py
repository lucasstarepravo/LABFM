import numpy as np
import math
from functions.nodes import neighbour_nodes
from functions.nodes import neighbour_nodes_kdtree
from scipy.spatial import cKDTree
from tqdm import tqdm
import time
import random


def monomial_power(polynomial):
    """

    :param polynomial:
    :return:
    """
    monomial_exponent = [(total_polynomial - i, i)
                         for total_polynomial in range(1, polynomial + 1)
                         for i in range(total_polynomial + 1)]
    return np.array(monomial_exponent)


def calc_scaling_vector(monomial_exponent, h):
    """

    :param monomial_exponent:
    :param h:
    :return:
    """
    scaling_m = np.zeros((len(monomial_exponent), 1))
    index = 0
    for exp_x, exp_y in monomial_exponent:
        exp_h = -(exp_x + exp_y)
        scaling_m[index] = h ** exp_h
        index = index + 1
    return scaling_m


def calc_monomial(neigh_xy_d, polynomial, mon_power):
    neigh_xy_d = neigh_xy_d
    monomial = []
    for index, (x_dist, y_dist) in enumerate(neigh_xy_d):
        row = []
        for power_x, power_y in mon_power:
            temp_variable = (x_dist ** power_x * y_dist ** power_y) / (
                        math.factorial(power_x) * math.factorial(power_y))
            row.append(temp_variable)
        monomial.append(row)
    monomial = np.array(monomial)
    return monomial.T


def gaussian_rbf(neighbours_r, h):
    """

    :param neighbours_r:
    :param h:
    :return:
    """

    q = neighbours_r / h

    w_ji = (9 / math.pi) * math.exp(-9 * (q ** 2))

    return w_ji


def wendland_rbf(neighbours_r, h):

    q = neighbours_r/h

    w_ji = (78/(28 * math.pi))*(1-q/2) ** 8 * (4*q**3 + 6.35 * q ** 2 + 4 * q + 1)
    return float(w_ji)


def wendland_rbf_c2(neighbours_r, h):

    q = neighbours_r/h

    w_ji = ((7/(4*math.pi))*(2*q+1)*(1-.5*q)**4)
    return float(w_ji)

def calc_hp(exp_a, dist_xy, h):
    """
    This function gives the first 10 expansions of the Hermite polynomial
    :param dist_xy:
    :param exp_a:
    :param h:
    :return:
    """

    z = dist_xy / (h * (2 ** .5))

    if exp_a == 0:
        h = 1

    elif exp_a == 1:
        h = 2 * z

    elif exp_a == 2:
        h = 4 * z ** 2 - 2

    elif exp_a == 3:
        h = 8 * z ** 3 - 12 * z

    elif exp_a == 4:
        h = 16 * z ** 4 - 48 * z ** 2 + 12

    elif exp_a == 5:
        h = 32 * z ** 5 - 160 * z ** 3 + 120 * z

    elif exp_a == 6:
        h = 64 * z ** 6 - 480 * z ** 4 + 720 * z ** 2 - 120

    elif exp_a == 7:
        h = 128 * z ** 7 - 1344 * z ** 5 + 3360 * z ** 3 - 1680 * z

    elif exp_a == 8:
        h = 256 * z ** 8 - 3584 * z ** 6 + 13440 * z ** 4 - 13440 * z ** 2 + 1680

    elif exp_a == 9:
        h = 512 * z ** 9 - 9216 * z ** 7 + 48384 * z ** 5 - 80640 * z ** 3 + 30240 * z

    elif exp_a == 10:
        h = 1024 * z ** 10 - 23040 * z ** 8 + 161280 * z ** 6 - 403200 * z ** 4 + 302400 * z ** 2 - 30240

    else:
        raise ValueError("Invalid value for polynomial. The polynomial must be a polynomial that can be handled by "
                         "the simulation. Try lowering the value of the polynomial")

    return h


def calc_abf(neigh_r, neigh_xy, m_power, h):  # Needs to be written
    """

    :param m_power:
    :param neigh_xy:
    :param neigh_r:
    :param h:
    :return:
    """
    basis_func = []

    for index, (x_dist, y_dist) in enumerate(neigh_xy):
        row = []
        rbf = wendland_rbf_c2(neigh_r[index], h)
        for power_a, power_b in m_power:
            h_a = calc_hp(power_a, x_dist, h)
            h_b = calc_hp(power_b, y_dist, h)
            two_power = 2 ** (power_a + power_b)
            temp_variable = rbf / (two_power ** .5) * h_a * h_b
            row.append(temp_variable)
        basis_func.append(row)
    return np.array(basis_func)


def calc_m(basis_func, monomial):
    """

    :param monomial:
    :param basis_func:
    :return:
    """
    m_matrix = np.zeros((basis_func.shape[1], monomial.shape[0]))
    for j in range(len(basis_func)):
        m_matrix = m_matrix + np.outer(monomial[:,j], basis_func[j])
    return m_matrix


def pointing_v(polynomial, d):
    """

    :param polynomial:
    :param d:
    :return:
    """

    if d not in ["x", "y", "Laplace"]:
        raise ValueError("The valid_string argument must be 'x', 'y', or 'Laplace'")

    n = int((polynomial ** 2 + 3 * polynomial) / 2)
    cd = np.zeros((n, 1))
    if d == 'x':
        cd[0] = 1
    elif d == 'y':
        cd[1] = 1
    elif d == 'Laplace':
        cd[2] = 1
        cd[4] = 1
    return cd


def add_uniform_noise(vector, percentage):
    # Calculate the noise bounds based on the maximum magnitude in the vector
    pointing_magnitude = max(np.abs(vector))
    noise_bounds = pointing_magnitude * (percentage / 100.0)

    # Generate uniform noise within the specified bounds for each entry
    # The size of the noise array matches the size of the input vector
    noise = np.random.uniform(-noise_bounds, noise_bounds, size=vector.shape)
    return noise


def psi_noise_uniform(psi_vector, absolute_error):
    noise = np.random.uniform(-abs(psi_vector)*absolute_error, abs(psi_vector)*absolute_error)
    return noise


def cd_noise_normal(psi_vector, mean_error, std_deviation):
    # Generate normal distribution noise with the specified mean and standard deviation
    noise = np.random.normal(mean_error, std_deviation, size=psi_vector.shape)
    noise = np.random.choice([-1, 1])*noise
    return noise


def calc_weights(coordinates, polynomial, h, total_nodes):
    """

    :param coordinates:
    :param polynomial:
    :param h:
    :return:
    """

    weights_x = {}
    weights_y = {}
    weights_laplace = {}
    neigh_coor_dict = {}

    monomial_exponent = monomial_power(polynomial)
    scaling_vector = calc_scaling_vector(monomial_exponent, h)
    cd_x = pointing_v(polynomial, 'x')
    cd_x = cd_x * scaling_vector

    #cd_x = cd_x + add_uniform_noise(cd_x, 0.0001)
    cd_y = pointing_v(polynomial, 'y')
    cd_y = cd_y * scaling_vector
    #cd_y = cd_y + add_uniform_noise(cd_y, 0.0001)
    cd_laplace = pointing_v(polynomial, 'Laplace')
    cd_laplace = cd_laplace * scaling_vector
    #cd_laplace = cd_laplace + add_uniform_noise(cd_laplace, 0.0001)
    tree = cKDTree(coordinates)
    total_laplace_time = 0

    for ref_x, ref_y in tqdm(coordinates, desc="Calculating Weights for " + str(total_nodes) + ", " + str(polynomial), ncols=100):
        if ref_x > 1 or ref_x < 0 or ref_y > 1 or ref_y < 0:
            continue
        else:
            ref_node            = (ref_x, ref_y)
            # neigh_r_d, neigh_xy_d, neigh_coor_dict[ref_node] = neighbour_nodes(coordinates, ref_node, h)
            neigh_r_d, neigh_xy_d, neigh_coor_dict[ref_node] = neighbour_nodes_kdtree(coordinates, ref_node, h, tree,
                                                                                      max_neighbors=None)
            monomial            = calc_monomial(neigh_xy_d, polynomial, monomial_exponent) * scaling_vector
            basis_func          = calc_abf(neigh_r_d, neigh_xy_d, monomial_exponent, h)
            m_matrix            = calc_m(basis_func, monomial)
            psi_x               = np.linalg.solve(m_matrix, cd_x + cd_noise_normal(cd_x, 0.001, 0.001)* scaling_vector)
            psi_y               = np.linalg.solve(m_matrix, cd_y + cd_noise_normal(cd_y, 0.001, 0.001)* scaling_vector)
            psi_laplace         = np.linalg.solve(m_matrix, cd_laplace + cd_noise_normal(cd_laplace, 0.000001, 0.000001)*max(scaling_vector))
            psi_x = psi_x# + psi_noise_normal(psi_x, 0.001, 0.001)
            psi_y = psi_y# + psi_noise_normal(psi_y, 0.001, 0.001)
            psi_laplace = psi_laplace# + psi_noise_normal(psi_laplace, 0.001, 0.001)
            node_weight_x       = basis_func @ psi_x
            node_weight_y       = basis_func @ psi_y
            start_time = time.time()  # Start timing
            node_weight_laplace = basis_func @ psi_laplace
            end_time = time.time()  # End timing
            total_laplace_time += (end_time - start_time)  # Accumulate time taken for this iteration
            weights_x[ref_node] = node_weight_x #+ psi_noise(node_weight_x, 0.001)
            weights_y[ref_node] = node_weight_y #+ psi_noise(node_weight_y, 0.001)
            weights_laplace[ref_node] = node_weight_laplace #+ psi_noise(node_weight_laplace, 0.001)

    print("Total time taken for Laplace calculation: ", total_laplace_time)
    return weights_x, weights_y, weights_laplace, neigh_coor_dict


def calc_l2(test_function, derivative):
    if derivative not in ['dtdx', 'dtdy', 'Laplace']:
        raise ValueError("Invalid derivative type")

    if derivative == 'dtdx':
        dt_analy = test_function.dtdx_true
        dt_aprox = test_function.dtdx_approx
    elif derivative == 'dtdy':
        dt_analy = test_function.dtdy_true
        dt_aprox = test_function.dtdy_approx
    else:
        dt_analy = test_function.laplace_true
        dt_aprox = test_function.laplace_approx

    error = []
    norm = []


    for ref_node in dt_aprox:
        error.append((dt_analy[ref_node] - dt_aprox[ref_node]) ** 2)
        norm.append(dt_analy[ref_node] ** 2)

    l2 = np.sqrt(sum(error)) / np.sqrt(sum(norm))

    return l2
