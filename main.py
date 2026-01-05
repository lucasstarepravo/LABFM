from classes.simulation import run
from functions.plot import plot_convergence, plot_stability, plot_resolving_p, plot_resolving_p_real
import pickle as pk


# Kernel options:
# Quintic Spline: 'q_s'
# Wendland C2:    'wc2'
# GNN:            'gnn'
# LABFM:          [2,4,6,8]

plot_ls = [False,
           False,
           True]

bool_plot_stability   = plot_ls[0]
bool_plot_convergence = plot_ls[1]
bool_plot_resolving_p = plot_ls[2]
idx_to_stability = 0
idx_to_res_power = 0

if __name__ == '__main__':
    total_nodes_list = [100] * 2
    kernel_list =  ['gnn']
    results = run(total_nodes_list, kernel_list)

# Plot stability of operator
if bool_plot_stability:
    plot_stability(results,
                   kernel=kernel_list[idx_to_stability],
                   resolution=total_nodes_list[idx_to_stability],
                   diff_operator='dx',
                   save=True)

if bool_plot_convergence:
    plot_convergence(results,
                     'dx',
                     size=20,
                     save=True,
                     show_legend=True)
    #plot_convergence(results, 'dy')
    #plot_convergence(results, 'laplace')
#import pickle
#with open('results_p2_n25.pk', 'wb') as f:
#    pickle.dump(results, f)
if bool_plot_resolving_p:
    plot_resolving_p_real(results, use_inset=False, zoom_y=True, save=True,
                          use_inset_x=True)