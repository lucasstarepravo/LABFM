from classes.simulation import run
from functions.plot import plot_convergence
import dill as pickle


if __name__ == '__main__':
    total_nodes_list = [10, 20, 50, 100, 200, 10, 20, 50, 100, 200, 10, 20, 50, 100, 200, 10, 20, 50, 100, 200]
    polynomial_list = [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8]
    results = run(total_nodes_list, polynomial_list)


#with open('results.dill', 'wb') as f:
#    pickle.dump(results, f)
plot_convergence(results, 'dtdx')
plot_convergence(results, 'dtdy')
plot_convergence(results, 'Laplace')