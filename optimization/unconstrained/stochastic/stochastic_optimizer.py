import itertools

import numpy as np

from ml.neural_network.initializers import random_uniform
from optimization.optimizer import Optimizer
from utils import iter_mini_batches


class StochasticOptimizer(Optimizer):
    def __init__(self, f, wrt=random_uniform, step_rate=0.01, momentum_type='none', momentum=0.9,
                 batch_size=None, eps=1e-6, max_iter=1000, verbose=False, plot=False):
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

        super().__init__(f, wrt, eps, max_iter, verbose, plot)
        if not np.isscalar(step_rate):
            raise ValueError('step_rate is not a real scalar')
        if not step_rate > 0:
            raise ValueError('step_rate must be > 0')
        self.step_rate = step_rate
        if not np.isscalar(momentum):
            raise ValueError('momentum is not a real scalar')
        if not momentum > 0:
            raise ValueError('momentum must be > 0')
        self.momentum = momentum
        if momentum_type not in ('standard', 'nesterov', 'none'):
            raise ValueError(f'unknown momentum type {momentum_type}')
        self.momentum_type = momentum_type
        if momentum_type in ('standard', 'nesterov'):
            self.step = 0
        self.args = (itertools.repeat(f.args()) if batch_size is None else
                     (i for i in iter_mini_batches(f.args(), batch_size)))

    def minimize(self):
        raise NotImplementedError
