from functions.discrete_operator import calc_l2, calc_l2_qspline, calc_l2_gnn, calc_l2_all
from functions.nodes import create_nodes, calc_h
from functions.plot import plot_weights, show_neighbours
from functions.qspline import qspline_weights
from functions.wendland_c2 import wendlandc2_weights
from functions.discrete_operator import calc_weights, gnn_weights_lap, gnn_weights_deriv
from functions.test_function import (test_function, dif_analytical, laplace_phi, dif_do,
                                     deriv_sph, lap_sph, dif_gnn, lap_moris)


class LABFM_Weights:
    def __init__(self, coordinates, polynomial, h, total_nodes):
        self.x, self.y, self.laplace, self._neigh_coor = calc_weights(coordinates, polynomial, h, total_nodes)

class WLandC2_Weights:
    def __init__(self, coordinates, h, total_nodes):
        (self.x, self.y, self.laplace, self.r,
         self.rho, self._neigh_coor, self._neigh_r, self._neigh_xy) = wendlandc2_weights(coordinates, h, total_nodes)

class QSPline_Weights:
    def __init__(self, coordinates, h, total_nodes):
        (self.x, self.y, self.laplace, # implement kernel radius deriv
         self.rho, self._neigh_coor, self._neigh_r) = qspline_weights(coordinates, h, total_nodes)

class GNN_Weights:
    def __init__(self, coordinates, h, total_nodes):
        self.laplace, self._neigh_coor = gnn_weights_lap(coordinates, h, total_nodes)
        self.x = gnn_weights_deriv(coordinates, h, total_nodes)
        '''Commenting out gnn dx'''

class TestFunction:
    def __init__(self,
                 coordinates,
                 labfm_w,
                 qspline_w,
                 wlandc2_w,
                 gnn_w,
                 h):

        self.surface_value = test_function(coordinates)
        self.dtdx_true = dif_analytical(coordinates, 'dtdx')
        self.dtdy_true = dif_analytical(coordinates, 'dtdy')
        self.laplace_true = laplace_phi(coordinates)

        self.dtdx_approx = dif_do(labfm_w, self.surface_value, 'dtdx')
        self.dtdy_approx = dif_do(labfm_w, self.surface_value, 'dtdy')
        self.laplace_labfm = dif_do(labfm_w, self.surface_value, 'Laplace')

        self.dtdx_qspline = deriv_sph(qspline_w, self.surface_value, 'dtdx')
        self.dtdy_qspline = deriv_sph(qspline_w, self.surface_value, 'dtdy')
        self.laplace_qspline = lap_sph(qspline_w, self.surface_value)

        self.dtdx_wc2 = deriv_sph(wlandc2_w, self.surface_value, 'dtdx')
        self.dtdy_wc2 = deriv_sph(wlandc2_w, self.surface_value, 'dtdy')
        self.laplace_wc2 = lap_moris(wlandc2_w, self.surface_value, 4*h)
        '''Commenting out gnn dx'''
        self.dtdx_gnn    = dif_gnn(gnn_w, self.surface_value, 'dtdx')
        self.laplace_gnn = dif_gnn(gnn_w, self.surface_value, 'Laplace')


class Simulation:
    def __init__(self, total_nodes, polynomial):
        # global variables used for all approximations
        self.s              = 1.0 / (total_nodes - 1)
        self.h              = calc_h(self.s, polynomial)
        self.coordinates    = create_nodes(total_nodes, self.s, polynomial)

        # Computing weights
        self.gnn_w          = GNN_Weights(self.coordinates, self.h, total_nodes)
        self.labfm_w        = LABFM_Weights(self.coordinates, polynomial, self.h, total_nodes)
        self.qspline_w      = QSPline_Weights(self.coordinates, self.h, total_nodes)
        self.wlandc2_w      = WLandC2_Weights(self.coordinates, 4*self.h, total_nodes)

        # Computing approximation of differential operators
        self.test_function  = TestFunction(self.coordinates, self.labfm_w,
                                           self.qspline_w, self.wlandc2_w,
                                           self.gnn_w, self.h)

        # Computing error of approximations
        self.dtdx_l2        = calc_l2(self.test_function, 'dtdx')
        self.dtdy_l2        = calc_l2(self.test_function, 'dtdy')
        self.laplace_l2     = calc_l2_all(self.test_function.laplace_labfm,
                                          self.test_function.laplace_true)

        self.dtdx_l2_qspline    = calc_l2_qspline(self.test_function, 'dtdx')
        self.dtdy_l2_qspline    = calc_l2_qspline(self.test_function, 'dtdy')
        self.laplace_l2_qspline = calc_l2_all(self.test_function.laplace_qspline,
                                              self.test_function.laplace_true)

        self.dtdx_l2_wc2       = calc_l2_all(self.test_function.dtdx_wc2,
                                              self.test_function.dtdx_true)
        self.dtdy_l2_wc2       = calc_l2_all(self.test_function.dtdy_wc2,
                                              self.test_function.dtdy_true)
        self.laplace_l2_wc2    = calc_l2_all(self.test_function.laplace_wc2,
                                              self.test_function.laplace_true)
        '''Commenting out gnn dx'''
        self.dtdx_l2_gnn        = calc_l2_all(self.test_function.dtdx_gnn,
                                              self.test_function.dtdx_true)
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
