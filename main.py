from classes.simulation import run
from functions.plot import plot_convergence
import dill as pickle


if __name__ == '__main__':
    total_nodes_list = [10, 10, 10, 10, 20, 20, 20, 20, 50, 50, 50, 50]
    polynomial_list = [2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8]
    results = run(total_nodes_list, polynomial_list)


#with open('results.dill', 'wb') as f:
#    pickle.dump(results, f)
plot_convergence(results, 'dtdx')
plot_convergence(results, 'dtdy')
plot_convergence(results, 'Laplace')