from classes.simulation import run
import dill as pickle


def plot_convergence(results, size=20):
    import numpy as np
    import matplotlib.pyplot as plt

    poly2 = {k: v for k, v in results.items() if k[1] == 2}
    poly4 = {k: v for k, v in results.items() if k[1] == 4}
    poly6 = {k: v for k, v in results.items() if k[1] == 6}

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

    plt.scatter(s2, dtdx2_l2, c='blue', label='Polynomial = 2', s=size)
    plt.scatter(s4, dtdx4_l2, c='red', label='Polynomial = 4', s=size)
    plt.scatter(s6, dtdx6_l2, c='green', label='Polynomial = 6', s=size)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Convergence of dtdx')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return


if __name__ == '__main__':
    total_nodes_list = [10, 10, 10]
    polynomial_list = [2, 4, 6]
    results = run(total_nodes_list, polynomial_list)


with open('results.dill', 'wb') as f:
    pickle.dump(results, f)

plot_convergence(results)