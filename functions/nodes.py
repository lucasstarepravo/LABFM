import numpy as np
from scipy.spatial import cKDTree
import random


def random_matrix(seed, shape, s):
    random.seed(seed)
    return np.array([[random.uniform(-s/6, s/6) for _ in range(shape[1])] for _ in range(shape[0])])


def calc_h(s, polynomial):
    """

    :param s:
    :param polynomial:
    :return:
    """
    if polynomial == 2:
        h = 1.5 * s
    elif polynomial == 4:
        h = 1.9 * s
    elif polynomial == 6:
        h = 2.3 * s
    elif polynomial == 8:
        h = 2.7 * s
    else:
        raise ValueError("The polynomial argument must be 2, 4, 6, or 8")

    return h


def create_nodes(total_nodes, s, polynomial):
    """
    :param total_nodes: is a scalar that states the number of nodes we would like to have inside the domain
    :param s: is a scalar that determines the average distance between points inside the comp. stencil
    :return:
    coordinates: is a numpy array that contains all coordinates of nodes, the first and second column contain the x and
    y coordinates, respectively. It is possible to access the ith node with "coordinates[i]"
    """
    delta = 1.0 / (total_nodes - 1)  # Determine the spacing delta between the points in the original domain
    h = calc_h(s, polynomial)
    n = int(2 / delta)  # Calculate the number of points to be added on each side
    x = np.linspace(-1, 2, total_nodes + total_nodes*2)  # Creates x coordinates with boundary
    y = np.linspace(-1, 2, total_nodes + total_nodes*2)  # Creates y coordinates with boundary

    X, Y = np.meshgrid(x, y)  # Create a 2D grid of x and y coordinates

    # Perturb the coordinates
    shift_x = random_matrix(1, X.shape, s)
    shift_y = random_matrix(2, Y.shape, s)
    X = X + shift_x
    Y = Y + shift_y

    # Stack the perturbed coordinates
    coordinates = np.column_stack((X.ravel(), Y.ravel()))
    coordinates = np.around(coordinates, 15)

    return coordinates


def neighbour_nodes(coordinates, ref_node, h, max_neighbors=None):
    neigh_r_d = []
    neigh_xy_d = []
    neigh_coor = []

    for index, (x_j, y_j) in enumerate(coordinates):
        distance = ((x_j - ref_node[0]) ** 2 + (y_j - ref_node[1]) ** 2) ** 0.5
        if distance <= 2 * h:
            neigh_r_d.append(distance)
            neigh_xy_d.append([x_j - ref_node[0], y_j - ref_node[1]])
            neigh_coor.append([x_j, y_j])

    # Combine lists into a single list of tuples for sorting
    combined_list = list(zip(neigh_r_d, neigh_xy_d, neigh_coor))

    # Sort the combined list by radial distance
    sorted_combined_list = sorted(combined_list, key=lambda x: x[0])

    # If max_neighbors is specified and less than the number of found neighbors, slice the lists
    if max_neighbors is not None and max_neighbors < len(sorted_combined_list):
        sorted_combined_list = sorted_combined_list[:max_neighbors]

    # Unzip the potentially sliced list into individual lists
    neigh_r_d, neigh_xy_d, neigh_coor = zip(*sorted_combined_list) if sorted_combined_list else ([], [], [])

    # Convert lists to numpy arrays for consistency with your return statement
    return np.array(list(neigh_r_d)), np.array(list(neigh_xy_d)), np.array(list(neigh_coor))


def neighbour_nodes_kdtree(coordinates, ref_node, h, tree, max_neighbors=1000):
    # Query the tree for points within a radius of h from the reference node
    indices = tree.query_ball_point(ref_node, h)
    max_neighbors = min(max_neighbors, len(indices))

    all_distances = np.sqrt(np.sum((coordinates[indices] - ref_node) ** 2, axis=1))
    sorted_indices = np.argsort(all_distances)[:max_neighbors]
    indices = np.array(indices)[sorted_indices]

    # Extract neighbor coordinates based on the filtered/sorted indices
    neigh_coor = coordinates[indices]

    # Calculate displacements and distances
    displacements = neigh_coor - ref_node
    #distances = np.linalg.norm(displacements, axis=1)
    distances = np.sqrt(np.sum(displacements**2, axis=1))
    return distances, displacements, neigh_coor
