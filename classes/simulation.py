from classes.weight import Weights
from classes.test_function import TestFunction
from functions.discrete_operator import calc_l2
from functions.nodes import create_nodes, calc_h
from functions.plot import plot_weights, show_neighbours


class Simulation:
    def __init__(self, total_nodes, polynomial):
        self.s             = 1.0 / (total_nodes - 1)
        self.h             = calc_h(self.s, polynomial)
        self.coordinates   = create_nodes(total_nodes, self.s, polynomial)
        self.weights       = Weights(self.coordinates, polynomial, self.h, total_nodes)
        self.test_function = TestFunction(self.coordinates, self.weights)
        self.dtdx_l2       = calc_l2(self.test_function, 'dtdx')
        self.dtdy_l2       = calc_l2(self.test_function, 'dtdy')
        self.laplace_l2    = calc_l2(self.test_function, 'Laplace')

    def plot_neighbours(self, size=8):
        return show_neighbours(self.coordinates, self.weights, size, save=False)

    def plot_weights(self, size=80, derivative='dtdx'):
        return plot_weights(self.coordinates, self.weights, size, derivative)


def run(total_nodes_list, polynomial_list):
    result = {}
    for total_nodes, polynomial in zip(total_nodes_list, polynomial_list):
        sim = Simulation(total_nodes, polynomial)
        result[(total_nodes, polynomial)] = sim  # .dtdx_l2, sim.dtdy_l2, sim.laplace_l2
    return result
