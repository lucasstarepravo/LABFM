import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from scipy.spatial import cKDTree
from tqdm import tqdm
import random
from functions.plot import plot_nodes



def random_matrix(seed: float, shape: tuple, s: float, noise: float) -> NDArray:
    random.seed(seed)
    noise = noise / 2
    return np.array([[random.uniform(-s*noise, s*noise) for _ in range(shape[1])] for _ in range(shape[0])])



def calc_h(s: float, kernel: int | str) -> float:
    # This determines the smoothing scale

    if isinstance(kernel, int):
        if   kernel <= 2: h = 1.5 * s
        elif kernel <= 4: h = 1.9 * s
        elif kernel <= 6: h = 2.3 * s
        elif kernel <= 8: h = 2.7 * s
        return h
    elif isinstance(kernel, str):
        if kernel == 'wc2': h = 1.5 * s
        elif kernel == 'q_s': h = 1.5 * s
        return h



def create_nodes(total_nodes: int, s: float, h: float,
                  shift_iterations: int = 100, shift_fraction: float = 0.1) -> Tuple[NDArray, int]:
    """
    :param total_nodes: is a scalar that states the number of nodes we would like to have inside the domain
    :param s: is a scalar that determines the average distance between points inside the comp. stencil
    :return:
    coordinates: is a numpy array that contains all coordinates of nodes, the first and second column contain the x and
    y coordinates, respectively. It is possible to access the ith node with "coordinates[i]"
    """
    delta = 1.0 / (total_nodes - 1)  # Determine the spacing delta between the points in the original domain
    two_times_influence_radius = 2 * h
    n = int(two_times_influence_radius / delta)  # Calculate the number of points to be added on each side

    # change this to only create nodes to the specific size of h
    x = np.linspace(-0.5 - two_times_influence_radius,
                    0.5 + two_times_influence_radius, total_nodes + n * 2)  # Creates x coordinates with boundary
    y = np.linspace(-0.5 - two_times_influence_radius,
                    0.5 + two_times_influence_radius, total_nodes + n * 2)  # Creates y coordinates with boundary

    X, Y = np.meshgrid(x, y)  # Create a 2D grid of x and y coordinates

    noise = 1.0
    # Perturb the coordinates
    shift_x = random_matrix(1, X.shape, s, noise)
    shift_y = random_matrix(2, Y.shape, s, noise)
    X = X + shift_x
    Y = Y + shift_y

    # Stack the perturbed coordinates
    coordinates = np.column_stack((X.ravel(), Y.ravel()))
    coordinates = np.around(coordinates, 15)

    nodes_in_domain = 0
    for ref_x, ref_y in coordinates:
        if ref_x > 0.5 or ref_x < -0.5 or ref_y > 0.5 or ref_y < -0.5: continue
        nodes_in_domain += 1

    shift_iterations = 0
    if shift_iterations > 0:
        in_domain = ((coordinates[:, 0] >= -0.5) & (coordinates[:, 0] <= 0.5) &
                     (coordinates[:, 1] >= -0.5) & (coordinates[:, 1] <= 0.5))
        interior = shift_nodes(coordinates[in_domain], 0.5, h, shift_iterations, shift_fraction)
        coordinates = np.vstack([interior, build_mirror_layer(interior, 0.5, depth=2 * h)])
        nodes_in_domain = len(interior)


    return coordinates, nodes_in_domain


def neighbour_nodes_kdtree(coordinates, ref_node, h, tree, max_neighbors=1000):
    # Query the tree for points within a radius of h from the reference node
    indices = tree.query_ball_point(ref_node, h)
    max_neighbors = min(max_neighbors, len(indices))

    all_distances = np.sqrt(np.sum((coordinates[indices] - ref_node) ** 2, axis=1))
    sorted_indices = np.argsort(all_distances)[:max_neighbors]
    indices = np.array(indices)[sorted_indices]

    # Extract neighbor coordinates based on the filtered/sorted indices
    neigh_coor = coordinates[indices]

    # Calculate relative position and distances
    relative_position = neigh_coor - ref_node
    distances = np.sqrt(np.sum(relative_position**2, axis=1))
    return distances, relative_position, neigh_coor


def build_mirror_layer(interior: NDArray, half_width: float, depth: float) -> NDArray:
    """Reflect interior points within `depth` of a wall/corner across that wall/corner."""
    x, y = interior[:, 0], interior[:, 1]
    mirrors = []

    for sx in (1.0, -1.0):
        near_wall = sx * x > half_width - depth
        if near_wall.any():
            m = interior[near_wall].copy()
            m[:, 0] = sx * 2 * half_width - m[:, 0]
            mirrors.append(m)

    for sy in (1.0, -1.0):
        near_wall = sy * y > half_width - depth
        if near_wall.any():
            m = interior[near_wall].copy()
            m[:, 1] = sy * 2 * half_width - m[:, 1]
            mirrors.append(m)

    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            near_corner = (sx * x > half_width - depth) & (sy * y > half_width - depth)
            if near_corner.any():
                m = interior[near_corner].copy()
                m[:, 0] = sx * 2 * half_width - m[:, 0]
                m[:, 1] = sy * 2 * half_width - m[:, 1]
                mirrors.append(m)

    return np.vstack(mirrors) if mirrors else np.empty((0, 2))


def shifting_step(interior: NDArray, mirrors: NDArray, h: float, shift_fraction: float) -> NDArray:
    """One particle-shifting pass: nudge each interior node away from its close neighbours."""
    all_coords = np.vstack([interior, mirrors])
    tree = cKDTree(all_coords)
    support_radius = 2 * h
    shift_coeff = shift_fraction * h

    dr = np.zeros_like(interior)
    for idx, node in enumerate(interior):
        distances, relative_position, _ = neighbour_nodes_kdtree(all_coords, node, support_radius, tree)
        rij = -relative_position[1:]  # node - neighbour, skipping self (row 0)
        rad = distances[1:]
        q = rad / h
        weight = shift_coeff * (q / 2 - 1) / rad
        dr_tmp = (weight[:, None] * rij).sum(axis=0)
        mag = np.linalg.norm(dr_tmp)
        if mag > shift_coeff:
            dr_tmp *= shift_coeff / mag
        dr[idx] = dr_tmp

    return interior - dr


def shift_nodes(interior: NDArray, half_width: float, h: float, n_iter: int, shift_fraction: float) -> NDArray:
    """Iteratively relax `interior` nodes via particle shifting, with reflective mirrors for boundaries."""
    depth = 2 * h
    for _ in tqdm(range(n_iter), desc="Shifting nodes", ncols=100):
        mirrors = build_mirror_layer(interior, half_width, depth)
        interior = shifting_step(interior, mirrors, h, shift_fraction)
    return interior


def point_cloud_statistics(coordinates, h):
        tree = cKDTree(coordinates)

        mask = (coordinates[:, 0] >= -0.5) & (coordinates[:, 0] <= 0.5) & \
               (coordinates[:, 1] >= -0.5) & (coordinates[:, 1] <= 0.5)

        filtered_coordinates = coordinates[mask]

        distances, indexes = tree.query(filtered_coordinates, 2, 2)

        nn_dists = distances[:, 1] / h  # drop self-distance
        mu = np.mean(nn_dists)
        sigma = np.std(nn_dists)
        return mu, sigma, sigma / mu


def full_point_cloud_statistics(coordinates, h, r_max=2.0, dr=0.05):
    tree = cKDTree(coordinates)

    mask = (coordinates[:, 0] >= -0.5) & (coordinates[:, 0] <= 0.5) & \
           (coordinates[:, 1] >= -0.5) & (coordinates[:, 1] <= 0.5)

    filtered_coordinates = coordinates[mask]

    # Nearest-neighbor stats (existing)
    distances, indexes = tree.query(filtered_coordinates, 2, 2)
    nn_dists = distances[:, 1] / h
    mu = np.mean(nn_dists)
    sigma = np.std(nn_dists)

    # g(r)
    bins = np.arange(dr, r_max + dr, dr) * h  # physical distances
    pairs = tree.query_ball_point(filtered_coordinates, r_max * h)

    counts = np.zeros(len(bins) - 1)
    for i, neighbors in enumerate(pairs):
        dists = np.linalg.norm(
            coordinates[neighbors] - filtered_coordinates[i], axis=1
        )
        dists = dists[dists > 0]  # drop self
        counts += np.histogram(dists, bins=bins)[0]

    # Normalize: N_particles, annular area, number density
    n_filtered = len(filtered_coordinates)
    rho = len(coordinates) / ((coordinates[:, 0].max() - coordinates[:, 0].min()) *
                               (coordinates[:, 1].max() - coordinates[:, 1].min()))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    annular_area = 2 * np.pi * bin_centers * (bins[1] - bins[0])
    gr = counts / (n_filtered * annular_area * rho)

    return mu, sigma, sigma / mu, bin_centers / h, gr