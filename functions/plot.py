import numpy as np
import matplotlib.pyplot as plt


def show_neighbours(coordinates, weights, size):
    neigh_coor = weights._neigh_coor
    out_domain = []

    keys_array = np.array(list(neigh_coor.keys()))

    while True:
        sample = keys_array[np.random.choice(len(keys_array))]
        if 0 <= sample[0] <= 1 and 0 <= sample[1] <= 1:
            ref_node = sample
            break

    for x_i, y_i in coordinates:
        if x_i < 0 or x_i > 1 or y_i < 0 or y_i > 1:
            out_domain.append([x_i, y_i])

    neigh = np.array(neigh_coor[tuple(ref_node)])
    out_domain = np.array(out_domain)

    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', label='All Nodes', s=size)
    plt.scatter(out_domain[:, 0], out_domain[:, 1], c='grey', label='Out of Domain', s=size)
    plt.scatter(neigh[:, 0], neigh[:, 1], c='red', label='Neighbours', s=size)
    plt.scatter(ref_node[0], ref_node[1], c='yellow', label='Reference Node', s=size)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Nodes and Neighbours')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    return


def plot_weights(coordinates, weights, size, derivative):
    while True:
        sample = coordinates[np.random.choice(coordinates.shape[0])]
        if 0 <= sample[0] <= 1 and 0 <= sample[1] <= 1:
            ref_node = sample
            break

    ref_neigh = np.array(weights._neigh_coor[tuple(ref_node)])
    index_to_delete = np.where((ref_neigh == ref_node).all(axis=1))
    mask = ~np.all(ref_neigh == ref_node, axis=1)
    ref_neigh = ref_neigh[mask]

    if derivative == 'dtdx':
        ref_weights = weights.x[tuple(ref_node)]
    elif derivative == 'dtdy':
        ref_weights = weights.y[tuple(ref_node)]
    elif derivative == 'Laplace':
        ref_weights = weights.laplace[tuple(ref_node)]

    ref_weights = np.delete(ref_weights, index_to_delete[0])

    plt.scatter(ref_node[0], ref_node[1], c='black', label='Reference Node', s=size)
    plt.scatter(ref_neigh[:, 0], ref_neigh[:, 1], c=ref_weights, label='Neighbour nodes', s=size, cmap='brg')
    plt.colorbar()
    plt.legend()
    plt.show()
    return


def plot_convergence(results, derivative='dtdx', size=20):
    import numpy as np
    import matplotlib.pyplot as plt

    poly2 = {k: v for k, v in results.items() if k[1] == 2}
    poly4 = {k: v for k, v in results.items() if k[1] == 4}
    poly6 = {k: v for k, v in results.items() if k[1] == 6}
    poly8 = {k: v for k, v in results.items() if k[1] == 8}

    if derivative == 'dtdx':
        # DTDX
        dtdx2_l2 = []
        s2 = []
        for tup in list(poly2.keys()):
            s2.append(1 / tup[0])
            dtdx2_l2.append(poly2[tup].dtdx_l2)
        s2 = np.array(s2)
        dtdx2_l2 = np.array(dtdx2_l2)

        dtdx4_l2 = []
        s4 = []
        for tup in list(poly4.keys()):
            s4.append(1 / tup[0])
            dtdx4_l2.append(poly4[tup].dtdx_l2)
        s4 = np.array(s4)
        dtdx4_l2 = np.array(dtdx4_l2)

        dtdx6_l2 = []
        s6 = []
        for tup in list(poly6.keys()):
            s6.append(1 / tup[0])
            dtdx6_l2.append(poly6[tup].dtdx_l2)
        s6 = np.array(s6)
        dtdx6_l2 = np.array(dtdx6_l2)

        dtdx8_l2 = []
        s8 = []
        for tup in list(poly8.keys()):
            s8.append(1 / tup[0])
            dtdx8_l2.append(poly8[tup].dtdx_l2)
        s8 = np.array(s8)
        dtdx8_l2 = np.array(dtdx8_l2)

    elif derivative == 'dtdy':
        # DTDY
        dtdx2_l2 = []
        s2 = []
        for tup in list(poly2.keys()):
            s2.append(1 / tup[0])
            dtdx2_l2.append(poly2[tup].dtdy_l2)
        s2 = np.array(s2)
        dtdx2_l2 = np.array(dtdx2_l2)

        dtdx4_l2 = []
        s4 = []
        for tup in list(poly4.keys()):
            s4.append(1 / tup[0])
            dtdx4_l2.append(poly4[tup].dtdy_l2)
        s4 = np.array(s4)
        dtdx4_l2 = np.array(dtdx4_l2)

        dtdx6_l2 = []
        s6 = []
        for tup in list(poly6.keys()):
            s6.append(1 / tup[0])
            dtdx6_l2.append(poly6[tup].dtdy_l2)
        s6 = np.array(s6)
        dtdx6_l2 = np.array(dtdx6_l2)

        dtdx8_l2 = []
        s8 = []
        for tup in list(poly8.keys()):
            s8.append(1 / tup[0])
            dtdx8_l2.append(poly8[tup].dtdy_l2)
        s8 = np.array(s8)
        dtdx8_l2 = np.array(dtdx8_l2)
    elif derivative == 'Laplace':
        # LAPLACE
        dtdx2_l2 = []
        s2 = []
        for tup in list(poly2.keys()):
            s2.append(1 / tup[0])
            dtdx2_l2.append(poly2[tup].laplace_l2)
        s2 = np.array(s2)
        dtdx2_l2 = np.array(dtdx2_l2)

        dtdx4_l2 = []
        s4 = []
        for tup in list(poly4.keys()):
            s4.append(1 / tup[0])
            dtdx4_l2.append(poly4[tup].laplace_l2)
        s4 = np.array(s4)
        dtdx4_l2 = np.array(dtdx4_l2)

        dtdx6_l2 = []
        s6 = []
        for tup in list(poly6.keys()):
            s6.append(1 / tup[0])
            dtdx6_l2.append(poly6[tup].laplace_l2)
        s6 = np.array(s6)
        dtdx6_l2 = np.array(dtdx6_l2)

        dtdx8_l2 = []
        s8 = []
        for tup in list(poly8.keys()):
            s8.append(1 / tup[0])
            dtdx8_l2.append(poly8[tup].laplace_l2)
        s8 = np.array(s8)
        dtdx8_l2 = np.array(dtdx8_l2)
    else:
        raise ValueError("Invalid derivative type")

    s2, s4, s6, s8 = [np.array(data).flatten() for data in [s2, s4, s6, s8]]
    dtdx2_l2, dtdx4_l2, dtdx6_l2, dtdx8_l2 = [np.array(data).flatten() for data in
                                              [dtdx2_l2, dtdx4_l2, dtdx6_l2, dtdx8_l2]]


    # Creating the scatter plot and lines connecting the points
    plt.scatter(s2, dtdx2_l2, c='blue', label='Polynomial = 2', s=size)
    plt.plot(s2, dtdx2_l2, c='blue')  # Line connecting points for Polynomial = 2

    plt.scatter(s4, dtdx4_l2, c='red', label='Polynomial = 4', s=size)
    plt.plot(s4, dtdx4_l2, c='red')  # Line connecting points for Polynomial = 4

    plt.scatter(s6, dtdx6_l2, c='green', label='Polynomial = 6', s=size)
    plt.plot(s6, dtdx6_l2, c='green')  # Line connecting points for Polynomial = 6

    plt.scatter(s8, dtdx8_l2, c='black', label='Polynomial = 8', s=size)
    plt.plot(s8, dtdx8_l2, c='black')  # Line connecting points for Polynomial = 8

    # Labels, title, legend, grid
    plt.xlabel('s/H')
    plt.ylabel('L2 norm')
    plt.title('Convergence of ' + derivative)
    plt.legend()
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(True, which='major', linestyle='-', linewidth='0.5', color='black')  # Major grid
    plt.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray')  # Minor grid (subgrid)

    # Log scale for x and y axes
    plt.xscale('log')
    plt.yscale('log')

    # Setting x and y axis ticks to show every 10th base
    def set_log_ticks(axis):
        locator = plt.LogLocator(base=10.0, numticks=12)
        axis.set_major_locator(locator)
        axis.set_minor_locator(locator)

    set_log_ticks(plt.gca().xaxis)
    set_log_ticks(plt.gca().yaxis)

    # Display the plot
    plt.show()
    return


def plot_convergence2(results, derivative='dtdx', size=20, plot_sph=True):
    import random
    # Dictionary to hold the data for each polynomial degree
    poly_data = {}
    _poly_degree = set()

    # Dynamically populate poly_data based on available polynomial degrees in results
    for k, v in results.items():
        poly_degree = k[1]
        if poly_degree in [2, 4, 6, 8]:  # Check if the degree is one of the interest
            _poly_degree.add(poly_degree)
            if poly_degree not in poly_data:
                poly_data[poly_degree] = {'s': [], 'l2': []}
            s_value = 1 / k[0]
            l2_value = getattr(v, f'{derivative}_l2')
            poly_data[poly_degree]['s'].append(s_value)
            poly_data[poly_degree]['l2'].append(l2_value)

    if plot_sph:
        poly_data['SPH'] = {'s': [], 'l2': []}
        p = random.choice(tuple(_poly_degree))
        for k, v in results.items():
            poly_degree = k[1]
            if poly_degree in [p]:  # Check if the degree is one of the interest
                s_value = 1 / k[0]
                l2_value = getattr(v, f'{derivative}_l2_sph')
                poly_data['SPH']['s'].append(s_value)
                poly_data['SPH']['l2'].append(l2_value)

    # Plotting
    colors = {2: 'blue', 4: 'red', 6: 'green', 8: 'black', 'SPH': 'purple'}
    labels = {2: 'Polynomial = 2', 4: 'Polynomial = 4', 6: 'Polynomial = 6', 8: 'Polynomial = 8',
              'SPH': 'Quintic Spline'}

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
        locator = plt.LogLocator(base=10.0, numticks=12)
        axis.set_major_locator(locator)
        axis.set_minor_locator(locator)

    set_log_ticks(plt.gca().xaxis)
    set_log_ticks(plt.gca().yaxis)

    plt.show()
