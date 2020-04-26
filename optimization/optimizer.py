import numpy as np

from ml.neural_network.initializers import random_uniform
from optimization.optimization_function import OptimizationFunction


class Optimizer:

    def __init__(self, f, wrt=random_uniform, eps=1e-6, max_iter=1000, verbose=False, plot=False):
        """

        :param f:        the objective function.
        :param wrt:      ([n x 1] real column vector): the point where to start the algorithm from.
        :param eps:      (real scalar, optional, default value 1e-6): the accuracy in the stopping
                         criterion: the algorithm is stopped when the norm of the gradient is less
                         than or equal to eps.
        :param max_iter: (integer scalar, optional, default value 1000): the maximum number of iterations.
        :param verbose:  (boolean, optional, default value False): print details about each iteration
                         if True, nothing otherwise.
        :param plot:     (boolean, optional, default value False): plot the function's surface and its contours
                         if True and the function's dimension is 2, nothing otherwise.
        """
        if not isinstance(f, OptimizationFunction):
            raise TypeError('f is not an optimization function')
        self.f = f
        if callable(wrt):
            self.wrt = wrt(f.n)
        elif not np.isrealobj(wrt):
            raise ValueError('x not a real vector')
        else:
            self.wrt = np.asarray(wrt, dtype=float)
        self.n = self.wrt.size
        if not np.isscalar(eps):
            raise ValueError('eps is not a real scalar')
        if not eps > 0:
            raise ValueError('eps must be > 0')
        self.eps = eps
        if not np.isscalar(max_iter):
            raise ValueError('max_iter is not an integer scalar')
        if not max_iter > 0:
            raise ValueError('max_iter must be > 0')
        self.max_iter = max_iter
        self.iter = 0
        self.verbose = verbose
        self.plot = plot

    def minimize(self):
        raise NotImplementedError
