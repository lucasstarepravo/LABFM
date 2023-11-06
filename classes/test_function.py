from functions.test_function import test_function, dif_analytical, laplace_phi, dif_do
from functions.local_test import test_function2, dif_analytical2, dif_do2


class TestFunction:
    def __init__(self, coordinates, weights):
        self.surface_value = test_function(coordinates)
        self.dtdx_true = dif_analytical(coordinates, 'dtdx')
        self.dtdy_true = dif_analytical(coordinates, 'dtdy')
        self.laplace_true = laplace_phi(coordinates)
        self.dtdx_approx = dif_do(weights, self.surface_value, 'dtdx')
        self.dtdy_approx = dif_do(weights, self.surface_value, 'dtdy')
        self.laplace_approx = dif_do(weights, self.surface_value, 'Laplace')


class local_TestFunction:
    def __init__(self, coordinates, weights, derivative):
        self.surface_value, self._neigh_value = test_function2(coordinates, weights, derivative)
        if derivative == 'dtdx':
            self.true = dif_analytical2(coordinates, 'dtdx')
            self.approx = dif_do2(weights, self.surface_value, self._neigh_value, 'dtdx')
        elif derivative == 'dtdy':
            self.true = dif_analytical2(coordinates, 'dtdy')
            self.approx = dif_do2(weights, self.surface_value, self._neigh_value,'dtdy')
        elif derivative == 'Laplace':
            self.true = dif_analytical2(coordinates, 'Laplace')
            self.approx = dif_do2(weights, self.surface_value, self._neigh_value, 'Laplace')
