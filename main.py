from classes.simulation import run
from functions.plot import plot_convergence, plot_stability, plot_resolving_p
import pickle as pk


# Kernel options:
# Quintic Spline: 'q_s'
# Wendland C2:    'wc2'
# GNN:            'gnn'
# LABFM:          [2,4,6,8]

plot_ls = [True,
           False,
           False]

bool_plot_stability   = plot_ls[0]
bool_plot_convergence = plot_ls[1]
bool_plot_resolving_p = plot_ls[2]
idx_to_stability = 0
idx_to_res_power = 2

if __name__ == '__main__':
    total_nodes_list = [50]
    kernel_list =  ['wc2']
    results = run(total_nodes_list, kernel_list)

# Plot stability of operator
if bool_plot_stability:
    plot_stability(results,
                   kernel=kernel_list[idx_to_stability],
                   resolution=total_nodes_list[idx_to_stability],
                   diff_operator='dx')

if bool_plot_convergence:
    plot_convergence(results,
                     'dx',
                     size=20)
    #plot_convergence(results, 'dy')
    #plot_convergence(results, 'laplace')


if bool_plot_resolving_p:
    plot_resolving_p(results,
                     kernel=kernel_list[idx_to_res_power],
                     resolution=total_nodes_list[idx_to_res_power])