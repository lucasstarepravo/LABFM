from classes.simulation import run
from functions.plot import plot_convergence, plot_stability
import pickle as pk


# Kernel options:
# Quintic Spline: 'quintic_s'
# Wendland C2:    'wc2'
# GNN:            'gnn'
# LABFM:          [2,4,6,8]

bool_plot_stability = True
bool_plot_convergence = False
idx_to_stability = 3

if __name__ == '__main__':
    total_nodes_list = [10, 20, 50, 100]
    kernel_list = ['quintic_s'] * 4
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
