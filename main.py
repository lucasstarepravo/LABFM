from classes.simulation import run
from functions.plot import plot_convergence, plot_stability
import pickle as pk


# Kernel options:
# Quintic Spline: 'quintic_s'
# Wendland C2:    'wc2'
# GNN:            'gnn'
# LABFM:          [2,4,6,8]

plot_ls = [False, True]

bool_plot_stability   = plot_ls[0]
bool_plot_convergence = plot_ls[1]
idx_to_stability = 0

if __name__ == '__main__':
    total_nodes_list = [10, 20, 50, 100, 200] * 3
    kernel_list = ['wc2'] * 5 + ['quintic_s'] * 5 + [2] * 5
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
                   kernel=kernel_list[idx_to_stability],
                   resolution=total_nodes_list[idx_to_stability],
                   diff_operator='dx')
