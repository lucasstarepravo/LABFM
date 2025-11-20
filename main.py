from classes.simulation import run
from functions.plot import plot_convergence, plot_stability
import pickle as pk


# Kernel options:
# Quintic Spline: 'quintic_s'
# Wendland C2:    'wc2'
# GNN:            'gnn'
# LABFM:          [2,4,6,8]

bool_plot_stability = False
bool_plot_convergence = True

if __name__ == '__main__':
    total_nodes_list = [5, 10, 20, 50, 100]
    kernel_list = ['gnn'] * 5
    results = run(total_nodes_list, kernel_list)

if bool_plot_convergence:
    plot_convergence(results,
                     'dx',
                     size=20)
    #plot_convergence(results, 'dy')
    #plot_convergence(results, 'laplace')

# Plot stability of operator
if bool_plot_stability:
    plot_stability(results,
                   kernel=kernel_list[0],
                   resolution=total_nodes_list[0],
                   diff_operator='dx')
