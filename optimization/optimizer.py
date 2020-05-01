import numpy as np

from ml.neural_network.initializers import random_uniform
from optimization.optimization_function import OptimizationFunction


class Optimizer:

    def __init__(self, f, x=random_uniform, eps=1e-6, max_iter=1000,
                 callback=None, callback_args=(), verbose=False, plot=False):
        """

        :param f:        the objective function.
        :param x:        ([n x 1] real column vector): the point where to start the algorithm from.
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
        if callable(x):
            self.x = x(f.ndim)
        elif not np.isrealobj(x):
            raise ValueError('x not a real vector')
        else:
            self.x = np.asarray(x, dtype=np.float)
        self.f_x = np.nan
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
        self._callback = callback
        self.callback_args = callback_args
        self.verbose = verbose
        self.plot = plot

    def callback(self):
        if callable(self._callback):
            self._callback(self.x, self.f_x, *self.callback_args)

    def plot_step(self, fig, x, last_x, color='k'):
        p_xy = np.vstack((x, last_x)).T
        fig.axes[0].plot(p_xy[0], p_xy[1], [self.f.function(self.x), self.f.function(last_x)],
                         marker='.', color=color)
        fig.axes[1].quiver(p_xy[0, 0], p_xy[1, 0], p_xy[0, 1] - p_xy[0, 0], p_xy[1, 1] - p_xy[1, 0],
                           scale_units='xy', angles='xy', scale=1, color=color)

    def minimize(self):
        raise NotImplementedError
