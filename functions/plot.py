import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs


def plot_kernel(features, labels=None, alpha=0.6, size = 2):
    # flatten all arrays consistently
    x = features[:, :, 0].flatten()
    y = features[:, :, 1].flatten()
    if labels is not None:
        c = labels.flatten()

    plt.figure(figsize=(6, 6))
    if labels is not None:
        sc = plt.scatter(x, y, c=c, cmap='viridis', s=size, alpha=alpha)
    else:
        sc = plt.scatter(x, y, cmap='viridis', s=size, alpha=alpha)
    plt.xlabel('x distance')
    plt.ylabel('y distance')
    plt.title('Neighbour offsets coloured by target')
    plt.axis('equal')
    plt.colorbar(sc, label='Target value')
    plt.show()


def plot_stability(results: dict,
                   kernel: str | int,
                   resolution: int,
                   diff_operator: str
                   ) -> None:
    print('Computing eigenvalues')
    # (results, model_or_polynomial, resolution, diff_operator)

    if kernel not in [2, 4, 5, 8, 'gnn', 'wc2', 'q_s']:
        raise ValueError("Value of the kernel should either be a LABFM polynomial"
                         " in results one of the following kernels 'gnn', 'wc2' "
                         "'q_s'")


    if diff_operator not in ['dx', 'dy', 'laplace']:
        raise ValueError("Differential operator must be one of the following"
                         " 'dx', 'dy', or 'laplace'")

    # extract the correct results from variable results that has the desired resolution
    # and the kernel specified
    if (resolution, kernel) not in results.keys():
        raise ValueError('Kernel-resolution combination specified not in results')

    # extracting all attributes from object
    attrs = results[(resolution, kernel)]

    # extracting weights from attributes
    if diff_operator == 'dx': weights = attrs.x
    if diff_operator == 'dy': weights = attrs.y
    if diff_operator == 'laplace': weights = attrs.laplace

    h = attrs.h
    # extracting coordinates from attributes
    coor = attrs.coordinates
    neigh_coor = attrs._neigh_coor

    # surface value is not actually required, I only need the order of the position of the nodes i.e. coor
    A = np.zeros((len(coor), len(coor)))

    coord_to_idx = {tuple(x): i for i, x in enumerate(coor)}


    if isinstance(h, dict):
        for i in range(len(coor)):
            loc = coor[i]
            if tuple(loc) not in weights.keys(): continue

            for j in range(neigh_coor[tuple(loc)].shape[0]):
                n_j = neigh_coor[tuple(loc)][j]
                neigh_idx = coord_to_idx[tuple(n_j)]
                A[i, neigh_idx] = np.array(weights[tuple(loc)][j] * h[tuple(loc)])

            A[i, i] = 0
            A[i, i] = - np.sum(A[i, :])
    else:
        for i in range(len(coor)):
            loc = coor[i]
            if tuple(loc) not in weights.keys(): continue

            for j in range(neigh_coor[tuple(loc)].shape[0]):
                n_j = neigh_coor[tuple(loc)][j]
                neigh_idx = coord_to_idx[tuple(n_j)]
                A[i, neigh_idx] = np.array(weights[tuple(loc)][j]*h)

            A[i, i] = 0
            A[i, i] = - np.sum(A[i, :])

    A_masked = np.ma.masked_values(A, 0)
    A_sparse = csr_matrix(A)


    kernel_plot_title = {
        'q_s': 'Quintic Spline',
        'wc2': 'Wendland C2',
        'gnn': 'GNN',
        2: r'LABFM $2^{\mathrm{nd}}$ Order',
        4: r'LABFM $4^{\mathrm{th}}$ Order',
        6: r'LABFM $6^{\mathrm{th}}$ Order',
        8: r'LABFM $8^{\mathrm{th}}$ Order',
    }

    vals = np.linalg.eigvals(A)

    real_parts = vals.real
    imag_parts = vals.imag

    plt.figure(figsize=(7, 7))
    plt.scatter(real_parts, imag_parts, s=10)  # larger markers for readability

    plt.xlabel(r"$\mathcal{Re}\, (\lambda$)", fontsize=14)
    plt.ylabel(r"$\mathcal{Im}\, (\lambda$)", fontsize=14)
    plt.title(f"Spectrum of {kernel_plot_title[kernel]}", fontsize=16)

    plt.axhline(0)
    plt.axvline(0)
    plt.grid(True, linestyle="--", linewidth=0.6)

    plt.tight_layout()
    plt.show()


def plot_resolving_p(results, kernel, resolution, size=20):
    res_power = results[(resolution, kernel)].k

    y1 = res_power[:, 0]  # first column
    y2 = res_power[:, 1]  # second column
    x = res_power[:, 2]  # third column

    # (Optional) sort by x so lines look nice
    idx = np.argsort(x)
    x_sorted = x[idx]
    y1_sorted = y1[idx]
    y2_sorted = y2[idx]

    # ---------- Figure 1 ----------
    # y-axis: first column
    # x-axis: third column
    # plus diagonal: third column vs third column
    plt.figure()
    plt.plot(x_sorted, y1_sorted, 'o-', label='LABFM 2nd order')
    plt.plot(x_sorted, x_sorted, '--', label='spectral accuracy')
    plt.xlabel('k_hat')
    plt.ylabel('Real (k_eff)')
    plt.legend()
    plt.grid(True)

    # ---------- Figure 2 ----------
    # y-axis: second column
    # x-axis: third column
    plt.figure()
    plt.plot(x_sorted, y2_sorted, 'o-', label='col1 vs col2')
    plt.xlabel('khat')
    plt.ylabel('Im (k_eff)')
    plt.legend()
    plt.grid(True)

    plt.show()

    return


def plot_convergence(results, derivative='dx', size=20):
    import random
    # Dictionary to hold the data for each polynomial degree
    poly_data = {}
    _poly_degree = set()

    # Dynamically populate poly_data based on available polynomial degrees in results
    for k, v in results.items():
        poly_degree = k[1]
        if poly_degree in [2, 4, 6, 8, 'q_s', 'wc2', 'gnn']:  # Check if the degree is one of the interest
            _poly_degree.add(poly_degree)
            if poly_degree not in poly_data:
                poly_data[poly_degree] = {'s': [], 'l2': []}
            s_value = 1 / k[0]
            l2_value = getattr(v, f'{derivative}_l2')
            poly_data[poly_degree]['s'].append(s_value)
            poly_data[poly_degree]['l2'].append(l2_value)


    # Plotting
    colors = {2: 'blue', 4: 'red', 6: 'green', 8: 'black',
              'q_s': 'purple', 'gnn': 'orange', 'wc2': 'cyan'}
    labels = {2: 'Polynomial = 2', 4: 'Polynomial = 4', 6: 'Polynomial = 6', 8: 'Polynomial = 8',
              'q_s': 'Quintic Spline', 'gnn': 'GNN', 'wc2': 'Wendland C${2}$'}

    for poly_degree, data in poly_data.items():
        s = np.array(data['s'])
        l2 = np.array(data['l2']).flatten()  # Ensure l2 is flattened
        plt.scatter(s, l2, c=colors[poly_degree], label=labels[poly_degree], s=size)
        plt.plot(s, l2, c=colors[poly_degree])  # Line connecting points

    # Labels, title, legend, grid, and scales
    plt.xlabel('s/H')
    plt.ylabel('L2 norm')
    plt.title('Convergence of ' + derivative)
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.xscale('log')
    plt.yscale('log')

    # Adjust log ticks
    def set_log_ticks(axis):
        major_locator = LogLocator(base=10.0, numticks=12)
        minor_locator = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
        axis.set_major_locator(major_locator)
        axis.set_minor_locator(minor_locator)

    set_log_ticks(plt.gca().xaxis)
    set_log_ticks(plt.gca().yaxis)

    plt.show()
