import itertools
import warnings
from abc import ABC

import numpy as np
from sklearn.utils import shuffle

from ... import Optimizer


class StochasticOptimizer(Optimizer, ABC):

    def __init__(self,
                 f,
                 x=np.random.uniform,
                 step_size=0.01,
                 batch_size=None,
                 eps=1e-6,
                 epochs=1000,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        """

        :param f: the objective function.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :param eps: (real scalar, optional, default value 1e-6): the accuracy in the stopping
                    criterion: the algorithm is stopped when the norm of the gradient is less
                    than or equal to eps.
        :param epochs: (integer scalar, optional, default value 1000): the maximum number of iterations.
        :param verbose: (boolean, optional, default value False): print details about each iteration
                        if True, nothing otherwise.
        """

        super().__init__(f, x, eps, epochs, callback, callback_args, verbose)
        if not step_size > 0:
            raise ValueError('step_size must be > 0')
        self.step_size = step_size
        self.epochs = epochs
        self.epoch = 0
        self.shuffle = shuffle
        self.random_state = random_state
        self.step = 0

        if batch_size is None:
            self.batch_size = None
            self.batches = itertools.repeat(f.args())
        else:
            n_samples = len(f.args()[0])

            if batch_size < 1 or batch_size > n_samples:
                warnings.warn('Got `batch_size` less than 1 or larger than '
                              'sample size. It is going to be clipped.')
            self.batch_size = np.clip(batch_size, 1, n_samples)

            self.n_batches, rest = divmod(len(f.args()[0]), self.batch_size)
            if rest:
                self.n_batches += 1

            self.max_iter *= self.n_batches

            self.batches = (i for i in self.iter_mini_batches())

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
            idx = list(range(self.n_batches))
            while True:
                if self.shuffle:
                    shuffle(idx, random_state=self.random_state)
                for i in idx:
                    start = i * self.batch_size
                    stop = (i + 1) * self.batch_size
                    yield [param[slice(start, stop)] for param in self.f.args()]

    def is_batch_end(self):
        return (self.batch_size is None or self.batch_size == len(self.f.args()[0])
                or (self.iter and not self.iter % self.n_batches))

    def is_verbose(self):
        return self.verbose and not self.epoch % self.verbose

    def _print_header(self):
        if self.verbose:
            print('epoch\titer\t cost\t', end='')
            if self.f.f_star() < np.inf:
                print('\t gap\t\t rate', end='')
                self.prev_f_x = np.inf

    def _print_info(self):
        if self.is_batch_end() and self.is_verbose():
            print('\n{:4d}\t{:4d}\t{: 1.4e}'.format(self.epoch, self.iter, self.f_x), end='')
            if self.f.f_star() < np.inf:
                print('\t{: 1.4e}'.format(self.f_x - self.f.f_star()), end='')
                if self.prev_f_x < np.inf:
                    print('\t{: 1.4e}'.format((self.f_x - self.f.f_star()) /
                                              (self.prev_f_x - self.f.f_star())), end='')
                else:
                    print('\t\t', end='')
                self.prev_f_x = self.f_x


class StochasticMomentumOptimizer(StochasticOptimizer, ABC):

    def __init__(self,
                 f,
                 x=np.random.uniform,
                 step_size=0.01,
                 momentum_type='none',
                 momentum=0.9,
                 batch_size=None,
                 eps=1e-6,
                 epochs=1000,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        super().__init__(f, x, step_size, batch_size, eps, epochs, callback,
                         callback_args, shuffle, random_state, verbose)
        if momentum_type not in ('standard', 'nesterov', 'none'):
            raise ValueError(f'unknown momentum type {momentum_type}')
        self.momentum_type = momentum_type
        if not momentum > 0:
            raise ValueError('momentum must be > 0')
        self.momentum = momentum
