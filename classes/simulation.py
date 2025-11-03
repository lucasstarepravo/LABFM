from functions.discrete_operator import calc_l2, calc_l2_sph, calc_l2_gnn
from functions.nodes import create_nodes, calc_h
from functions.plot import plot_weights, show_neighbours
from functions.sph import sph_weights
from functions.discrete_operator import calc_weights, gnn_weights
from functions.test_function import test_function, dif_analytical, laplace_phi, dif_do, deriv_sph, lap_sph, dif_gnn


class TestFunction:
    def __init__(self, coordinates, labfm_w, sph_w, gnn_w):
        self.surface_value = test_function(coordinates)
        self.dtdx_true = dif_analytical(coordinates, 'dtdx')
        self.dtdy_true = dif_analytical(coordinates, 'dtdy')
        self.laplace_true = laplace_phi(coordinates)
        self.dtdx_approx = dif_do(labfm_w, self.surface_value, 'dtdx')
        self.dtdy_approx = dif_do(labfm_w, self.surface_value, 'dtdy')
        self.laplace_approx = dif_do(labfm_w, self.surface_value, 'Laplace')

        self.dtdx_sph = deriv_sph(sph_w, self.surface_value, 'dtdx')
        self.dtdy_sph = deriv_sph(sph_w, self.surface_value, 'dtdy')
        self.laplace_sph = lap_sph(sph_w, self.surface_value)

        self.laplace_gnn = dif_gnn(gnn_w, self.surface_value, 'Laplace')


class LABFM_Weights:
    def __init__(self, coordinates, polynomial, h, total_nodes):
        self.x, self.y, self.laplace, self._neigh_coor = calc_weights(coordinates, polynomial, h, total_nodes)


class SPH_Weights:
    def __init__(self, coordinates, h, total_nodes):
        (self.x, self.y, self.laplace,
         self.rho, self._neigh_coor, self._neigh_r) = sph_weights(coordinates, h, total_nodes)

class GNN_Weights:
    def __init__(self, coordinates, h, total_nodes):
        self.laplace, self._neigh_coor = gnn_weights(coordinates, h, total_nodes)


class Simulation:
    def __init__(self, total_nodes, polynomial):
        self.s              = 1.0 / (total_nodes - 1)
        self.h              = calc_h(self.s, polynomial)
        self.coordinates    = create_nodes(total_nodes, self.s, polynomial)
        self.gnn_w          = GNN_Weights(self.coordinates, self.h, total_nodes)
        self.labfm_w        = LABFM_Weights(self.coordinates, polynomial, self.h, total_nodes)
        self.sph_w          = SPH_Weights(self.coordinates, self.s, total_nodes) # implementing SPH weights

        self.test_function  = TestFunction(self.coordinates, self.labfm_w  , self.sph_w, self.gnn_w)
        self.dtdx_l2        = calc_l2(self.test_function, 'dtdx')
        self.dtdy_l2        = calc_l2(self.test_function, 'dtdy')
        self.laplace_l2     = calc_l2(self.test_function, 'Laplace')

        self.dtdx_l2_sph    = calc_l2_sph(self.test_function, 'dtdx')
        self.dtdy_l2_sph    = calc_l2_sph(self.test_function, 'dtdx')
        self.laplace_l2_sph = calc_l2_sph(self.test_function, 'dtdx')

        self.laplace_l2_gnn     =  calc_l2_gnn(self.test_function, 'Laplace')

    def plot_neighbours(self, size=8):
        return show_neighbours(self.coordinates, self.labfm_w, size)

    def plot_weights(self, size=80, derivative='dtdx'):
        return plot_weights(self.coordinates, self.labfm_w, size, derivative)


def run(total_nodes_list, polynomial_list):
    result = {}
    for total_nodes, polynomial in zip(total_nodes_list, polynomial_list):
        sim = Simulation(total_nodes, polynomial)
        result[(total_nodes, polynomial)] = sim
    return result
