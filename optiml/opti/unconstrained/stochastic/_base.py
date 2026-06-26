import itertools
import warnings
from abc import ABC
from collections.abc import Iterable

import numpy as np
from sklearn.utils import shuffle

from .schedules import constant
from ... import Optimizer


class StochasticOptimizer(Optimizer, ABC):
    """
    Abstract base class for stochastic (mini-batch) first-order optimizers.

    At each iteration the gradient is estimated on a mini batch of the data
    drawn from f.args() rather than on the whole sample, and the point is moved
    by a step controlled by a (possibly adaptive or scheduled) learning rate.
    The mini batches are produced by iterating over the data, optionally shuffled
    once per epoch, where an epoch is a full pass over all the samples.
    """

    def __init__(self,
                 f,
                 x=None,
                 step_size=0.01,
                 batch_size=None,
                 eps=1e-6,
                 tol=1e-8,
                 epochs=1000,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        """

        :param f:             the objective function.
        :param x:             ([n x 1] real column vector): the point where to start the algorithm from.
        :param step_size:     (real scalar > 0, callable or iterable, optional, default value 0.01): the
                              learning rate, i.e., the size of the step taken along the search direction.
                              It can be a positive scalar (kept constant for all iterations), a callable
                              with signature step_size(X_batch, y_batch) returning an iterator (e.g., one
                              of the schedules in schedules.py) or an iterable yielding the step size to
                              use at each iteration.
        :param batch_size:    (integer scalar or None, optional, default value None): the size of the mini
                              batches used to estimate the gradient. If None the full sample is used at
                              every iteration (i.e., plain batch gradient descent); otherwise it is clipped
                              to lie in [1, n_samples].
        :param eps:           (real scalar, optional, default value 1e-6): the accuracy in the stopping
                              criterion: the algorithm is stopped when the norm of the gradient is less
                              than or equal to eps.
        :param tol:           (real scalar, optional, default value 1e-8): the tolerance used in the
                              optimality conditions of the Lagrangian dual (when f is a Lagrangian dual
                              function), i.e., the algorithm is stopped when the variables or the
                              constraints change by less than tol.
        :param epochs:        (integer scalar, optional, default value 1000): the maximum number of epochs,
                              i.e., full passes over the whole sample, before the algorithm is stopped.
        :param callback:      (callable, optional, default value None): a function called at each iteration
                              with the optimizer instance (and callback_args) as arguments; it can raise
                              StopIteration to interrupt the optimization.
        :param callback_args: (tuple, optional, default value ()): additional positional arguments passed
                              to the callback at each call.
        :param shuffle:       (boolean, optional, default value True): whether to shuffle the order of the
                              mini batches at the beginning of each epoch.
        :param random_state:  (integer scalar or None, optional, default value None): seed for the random
                              number generator used to initialize x (when not provided) and to shuffle the
                              mini batches, for reproducibility.
        :param verbose:       (boolean or integer, optional, default value False): print details about each
                              iteration if True (or every `verbose` epochs if an integer), nothing otherwise.
        """
        super(StochasticOptimizer, self).__init__(f=f,
                                                  x=x,
                                                  eps=eps,
                                                  tol=tol,
                                                  max_iter=epochs,
                                                  callback=callback,
                                                  callback_args=callback_args,
                                                  random_state=random_state,
                                                  verbose=verbose)
        if not callable(step_size) and not isinstance(step_size, Iterable) and not step_size > 0:
            raise ValueError('step_size must be > 0 or a callable or an iterator')
        if isinstance(step_size, Iterable):  # wrap the iterable into a function to deal with mini batches, i.e., *args
            self.step_size = lambda *args: step_size
        elif callable(step_size):  # i.e., step_size(X_batch, y_batch)
            self.step_size = step_size
        else:
            self.step_size = lambda *args: constant(step_size)
        self.epochs = epochs
        self.epoch = 0
        self.shuffle = shuffle
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
                print('\t{: 1.4e}'.format((self.f_x - self.f.f_star()) /
                                          max(abs(self.f.f_star()), 1)), end='')
                if self.prev_f_x < np.inf:
                    print('\t{: 1.4e}'.format((self.f_x - self.f.f_star()) /
                                              max(abs(self.prev_f_x - self.f.f_star()), 1)), end='')
                else:
                    print('\t\t', end='')
                self.prev_f_x = self.f_x


class StochasticMomentumOptimizer(StochasticOptimizer, ABC):
    """
    Abstract base class for stochastic optimizers that support a momentum term.

    In addition to the plain stochastic step it keeps a velocity that accumulates
    an exponentially decaying fraction of the past steps, which damps oscillations
    and accelerates convergence along consistent descent directions. Both the
    classical heavy-ball (Polyak) momentum and Nesterov's accelerated momentum are
    supported, as selected by momentum_type.
    """

    def __init__(self,
                 f,
                 x=None,
                 step_size=0.01,
                 momentum_type='none',
                 momentum=0.9,
                 batch_size=None,
                 eps=1e-6,
                 tol=1e-8,
                 epochs=1000,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        """

        :param f:             the objective function.
        :param x:             ([n x 1] real column vector): the point where to start the algorithm from.
        :param step_size:     (real scalar > 0, callable or iterable, optional, default value 0.01): the
                              learning rate, i.e., the size of the step taken along the search direction
                              (see StochasticOptimizer for the accepted forms).
        :param momentum_type: (string in {'none', 'polyak', 'nesterov'}, optional, default value 'none'):
                              the kind of momentum to apply: 'none' disables momentum, 'polyak' uses the
                              classical heavy-ball momentum and 'nesterov' uses Nesterov's accelerated
                              momentum (the gradient is evaluated after the momentum jump).
        :param momentum:      (real scalar in [0, 1) or iterable, optional, default value 0.9): the momentum
                              factor, i.e., the fraction of the previous step retained in the current one.
                              It can be a scalar (kept constant) or an iterable yielding the value to use at
                              each iteration.
        :param batch_size:    (integer scalar or None, optional, default value None): the size of the mini
                              batches used to estimate the gradient; if None the full sample is used.
        :param eps:           (real scalar, optional, default value 1e-6): the accuracy in the stopping
                              criterion: the algorithm is stopped when the norm of the gradient is less
                              than or equal to eps.
        :param tol:           (real scalar, optional, default value 1e-8): the tolerance used in the
                              optimality conditions of the Lagrangian dual (when f is a Lagrangian dual).
        :param epochs:        (integer scalar, optional, default value 1000): the maximum number of epochs
                              before the algorithm is stopped.
        :param callback:      (callable, optional, default value None): a function called at each iteration
                              with the optimizer instance (and callback_args) as arguments; it can raise
                              StopIteration to interrupt the optimization.
        :param callback_args: (tuple, optional, default value ()): additional positional arguments passed
                              to the callback at each call.
        :param shuffle:       (boolean, optional, default value True): whether to shuffle the order of the
                              mini batches at the beginning of each epoch.
        :param random_state:  (integer scalar or None, optional, default value None): seed for the random
                              number generator, for reproducibility.
        :param verbose:       (boolean or integer, optional, default value False): print details about each
                              iteration if True (or every `verbose` epochs if an integer), nothing otherwise.
        """
        super(StochasticMomentumOptimizer, self).__init__(f=f,
                                                          x=x,
                                                          step_size=step_size,
                                                          batch_size=batch_size,
                                                          eps=eps,
                                                          tol=tol,
                                                          epochs=epochs,
                                                          callback=callback,
                                                          callback_args=callback_args,
                                                          shuffle=shuffle,
                                                          random_state=random_state,
                                                          verbose=verbose)
        if momentum_type not in ('polyak', 'nesterov', 'none'):
            raise ValueError(f'unknown momentum type {momentum_type}')
        self.momentum_type = momentum_type
        if not isinstance(momentum, Iterable) and not 0 <= momentum < 1:
            raise ValueError('momentum must be between 0 and 1 or an iterator')
        if isinstance(momentum, Iterable):
            self.momentum = momentum
        else:
            self.momentum = constant(momentum)
