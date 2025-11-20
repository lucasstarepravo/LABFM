from functions.labfm_operator import calc_l2, calc_l2_gnn
from functions.nodes import create_nodes, calc_h
from functions.qspline_operator import qspline_weights
from functions.wendland_c2_operator import wendlandc2_weights
from functions.labfm_operator import calc_weights
from functions.gnn_operator import gnn_weights
from functions.test_function import (test_function, dif_analytical, laplace_phi, dif_do,
                                     deriv_sph, lap_sph_standard, lap_moris)


class AbstractBaseClass:
    def __init__(self, total_nodes, h):
        # global variables used for all approximations
        self.s              = 1.0 / (total_nodes - 1)
        self.coordinates    = create_nodes(total_nodes, self.s, h)
        self.total_nodes    = total_nodes


    def test_function_method(self):
        # used to compute the true values of the surface and its differential fields
        self.surface_value = test_function(self.coordinates)
        self.dtdx_true     = dif_analytical(self.coordinates, 'dtdx')
        self.dtdy_true     = dif_analytical(self.coordinates, 'dtdy')
        self.laplace_true  = laplace_phi(self.coordinates)


    def approx_diff_op(self):
        # fix this function to only require the weights, surface value, and neigh_coor, no need for derivative string
        self.dtdx_approx    = dif_do(self.x, self.surface_value, self._neigh_coor)
        #self.dtdy_approx    = dif_do(self.y, self.surface_value, self._neigh_coor)
        self.laplace_approx = dif_do(self.laplace, self.surface_value, self._neigh_coor)


    def approx_diff_op_sph(self, moris_op=True):
        # fix this function to only require the weights, surface value, and neigh_coor, no need for derivative string
        self.dtdx_approx    = deriv_sph(self.x, self.surface_value, self._neigh_coor, self.rho) # make sure rho was already computed here
        self.dtdy_approx    = deriv_sph(self.y, self.surface_value, self._neigh_coor, self.rho)
        if moris_op:
            lap_sph = lap_moris
            args = (self._neigh_coor, self._neigh_r, self.x, self.y, self._neigh_xy, self.rho,
                    self.h, self.surface_value)
        else:
            lap_sph = lap_sph_standard
            args = (self._neigh_coor, self._neigh_r, self.laplace, self.rho, self.surface_value)
        self.laplace_approx = lap_sph(*args) # still need to get this to work


    def calc_l2(self):
        # to call calc_l2_all the first arg must be the weights of the
        self.dx_l2       = calc_l2(self.dtdx_approx, self.dtdx_true)
        #self.dy_l2       = calc_l2(self.dtdy_approx, self.dtdy_true)
        self.laplace_l2  = calc_l2(self.laplace_approx, self.laplace_true)



class LABFM(AbstractBaseClass):
    def __init__(self, polynomial, total_nodes):
        self.s = 1.0 / (total_nodes - 1)
        self.h = calc_h(self.s, polynomial)
        super().__init__(total_nodes, self.h)
        self.polynomial = polynomial
        (self.x,
         self.y,
         self.laplace,
         self._neigh_coor) = calc_weights(self.coordinates, self.polynomial, self.h, self.total_nodes)
        self.test_function_method()
        self.approx_diff_op()
        self.calc_l2()


class GNN(AbstractBaseClass):
    def __init__(self, total_nodes):
        # for gnn, I don't really care about h, I care about the number of neighbours
        # so I can theoretically set h equivalent ot max polynomial and in the discrete operator
        # I filter the number of neighbours I want
        self.s = 1.0 / (total_nodes - 1)
        self.h = calc_h(self.s, kernel=8)  # need to make sure that this is fine
        super().__init__(total_nodes, self.h)
        # join all weight computations of the GNN in one function
        (self.x,
         self.laplace,
         self._neigh_coor) = gnn_weights(self.coordinates, self.h, self.total_nodes)
        self.test_function_method()
        self.approx_diff_op()
        self.calc_l2()


class WLandC2(AbstractBaseClass):
    def __init__(self, total_nodes):
        self.s = 1.0 / (total_nodes - 1)
        self.h = 4 * self.s
        super().__init__(total_nodes, self.h)
        (self.x,
         self.y,
         self.laplace,
         self.r,
         self.rho,
         self._neigh_coor,
         self._neigh_r,
         self._neigh_xy) = wendlandc2_weights(self.coordinates, self.h, self.total_nodes)
        self.test_function_method()
        self.approx_diff_op_sph()
        self.calc_l2()


class QSPline(AbstractBaseClass):
    def __init__(self, total_nodes):
        self.s = 1.0 / (total_nodes - 1)
        self.h = 4 * self.s
        super().__init__(total_nodes, self.h)
        (self.x,
         self.y,
         self.laplace, # implement kernel radius deriv
         self.rho,
         self._neigh_coor,
         self._neigh_r) = qspline_weights(self.coordinates, self.h, self.total_nodes)
        self.test_function_method()
        self.approx_diff_op_sph(moris_op=False)
        self.calc_l2()



def run(total_nodes_list, kernel_list):
    result = {}
    for total_nodes, k in zip(total_nodes_list, kernel_list):

        # restructure the classes to have the weights in here as just .x .y .laplace
        args = (total_nodes,)
        if k in [2, 4, 6, 8]: kernel, args = LABFM, (k, total_nodes)
        elif k == 'quintic_s': kernel = QSPline
        elif k == 'wc2': kernel = WLandC2
        elif k == 'gnn': kernel = GNN
        else: raise ValueError(" kernel must be either polynomial order for labfm (2, 4, 6, 8), or one of "
                               "the following kernels 'wc2', 'quintic_s', 'gnn'")

        sim = kernel(*args)
        result[(total_nodes, k)] = sim
    return result
