import itertools
import random

import numpy as np

from ml.neural_network.initializers import random_uniform
from optimization.optimizer import Optimizer


class StochasticOptimizer(Optimizer):
    def __init__(self, f, x=random_uniform, step_size=0.01, momentum_type='none', momentum=0.9, batch_size=None,
                 eps=1e-6, epochs=1000, callback=None, callback_args=(), verbose=False, plot=False):
        """

        :param f: the objective function.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :param eps: (real scalar, optional, default value 1e-6): the accuracy in the stopping
                    criterion: the algorithm is stopped when the norm of the gradient is less
                    than or equal to eps.
        :param epochs: (integer scalar, optional, default value 1000): the maximum number of iterations.
        :param verbose: (boolean, optional, default value False): print details about each iteration
                        if True, nothing otherwise.
        :param plot: (boolean, optional, default value False): plot the function's surface and its contours
                     if True and the function's dimension is 2, nothing otherwise.
        """

        super().__init__(f, x, eps, epochs, callback, callback_args, verbose, plot)
        if not np.isscalar(step_size):
            raise ValueError('step_size is not a real scalar')
        if not step_size > 0:
            raise ValueError('step_size must be > 0')
        self.step_size = step_size
        if not np.isscalar(momentum):
            raise ValueError('momentum is not a real scalar')
        if not momentum > 0:
            raise ValueError('momentum must be > 0')
        self.momentum = momentum
        if momentum_type not in ('standard', 'nesterov', 'none'):
            raise ValueError(f'unknown momentum type {momentum_type}')
        self.momentum_type = momentum_type
        self.step = 0

        if batch_size is None:
            self.args = itertools.repeat(f.args())
        else:
            if f.args()[0].shape[0] != f.args()[1].shape[0]:
                raise ValueError('X and y have unequal lengths')

            if batch_size > f.args()[0].shape[0]:
                raise ValueError('batch_size must be less than the number of examples')

            n_batches, rest = divmod(len(f.args()[0]), batch_size)
            if rest:
                n_batches += 1

            self.batch_size = n_batches
            self.max_iter *= n_batches

            self.args = (i for i in self.iter_mini_batches())

    def iter_mini_batches(self):
        """Return an iterator that successively yields tuples containing aligned
        mini batches of size batch_size from sliceable objects given in f.args(), in
        random order without replacement.
        Because different containers might require slicing over different
        dimensions, the dimension of each container has to be givens as a list
        dims.
        :param: Xy: tuple of arrays to be sliced into mini batches in alignment with the others
        :param: batch_size: size of each batch
        :return: infinite iterator of mini batches in random order (without replacement)
        """

        while True:
            idx = list(range(self.batch_size))
            while True:
                random.shuffle(idx)
                for i in idx:
                    start = i * self.batch_size
                    stop = (i + 1) * self.batch_size
                    yield [param[slice(start, stop)] for param in self.f.args()]

    def minimize(self):
        raise NotImplementedError
