import numpy as np

from . import StochasticMomentumOptimizer


class RMSProp(StochasticMomentumOptimizer):
    """
    RMSProp for the minimization of the provided function f.

    It divides the learning rate of each coordinate by the square root of an
    exponentially decaying average of the squared gradients (the moving root mean
    square), so that the effective step size adapts to the recent magnitude of the
    gradients; an optional Polyak or Nesterov momentum can be applied on top.

    References
    ----------
    .. [1] Tieleman, T. & Hinton, G. (2012). Lecture 6.5 - RMSProp, COURSERA:
       Neural Networks for Machine Learning.
    """

    def __init__(self,
                 f,
                 x=None,
                 step_size=0.001,
                 momentum_type='none',
                 momentum=0.9,
                 batch_size=None,
                 eps=1e-6,
                 tol=1e-8,
                 epochs=1000,
                 decay=0.9,
                 offset=1e-8,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        """

        :param f:             the objective function.
        :param x:             ([n x 1] real column vector): the point where to start the algorithm from.
        :param step_size:     (real scalar > 0, callable or iterable, optional, default value 0.001): the
                              learning rate, i.e., the base size of the step taken along the search direction.
        :param momentum_type: (string in {'none', 'polyak', 'nesterov'}, optional, default value 'none'):
                              the kind of momentum applied on top of the RMSProp step.
        :param momentum:      (real scalar in [0, 1) or iterable, optional, default value 0.9): the momentum
                              factor, i.e., the fraction of the previous step retained in the current one.
        :param batch_size:    (integer scalar or None, optional, default value None): the size of the mini
                              batches used to estimate the gradient; if None the full sample is used.
        :param eps:           (real scalar, optional, default value 1e-6): the accuracy in the stopping
                              criterion: the algorithm is stopped when the norm of the gradient is less
                              than or equal to eps.
        :param tol:           (real scalar, optional, default value 1e-8): the tolerance used in the
                              optimality conditions of the Lagrangian dual (when f is a Lagrangian dual).
        :param epochs:        (integer scalar, optional, default value 1000): the maximum number of epochs
                              before the algorithm is stopped.
        :param decay:         (real scalar in [0, 1), optional, default value 0.9): the exponential decay
                              rate of the moving average of the squared gradients.
        :param offset:        (real scalar > 0, optional, default value 1e-8): a small constant added to the
                              denominator to avoid division by zero and improve numerical stability.
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
        super(RMSProp, self).__init__(f=f,
                                      x=x,
                                      step_size=step_size,
                                      momentum=momentum,
                                      momentum_type=momentum_type,
                                      batch_size=batch_size,
                                      eps=eps,
                                      tol=tol,
                                      epochs=epochs,
                                      callback=callback,
                                      callback_args=callback_args,
                                      shuffle=shuffle,
                                      random_state=random_state,
                                      verbose=verbose)
        if not 0 <= decay < 1:
            raise ValueError('decay has to lie in [0, 1)')
        self.decay = decay
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset
        self.moving_mean_squared = np.ones_like(self.x)

    def minimize(self):

        self._print_header()

        for batch in self.batches:

            if self.momentum_type == 'nesterov':
                step_m1 = self.step
                step1 = next(self.momentum) * step_m1
                self.x += step1

            self.f_x, self.g_x = self.f.function_jacobian(self.x, *batch)

            self._print_info()

            try:
                self.callback(batch)
            except StopIteration:
                break

            if self.is_batch_end():
                self.epoch += 1

            if self.epoch >= self.epochs:
                self.status = 'stopped'
                break

            # compute search direction
            d = -self.g_x

            self.moving_mean_squared = self.decay * self.moving_mean_squared + (1. - self.decay) * self.g_x ** 2
            step2 = next(self.step_size(*batch)) * d / np.sqrt(self.moving_mean_squared + self.offset)

            if self.momentum_type == 'polyak':

                step_m1 = self.step
                self.step = next(self.momentum) * step_m1 + step2
                self.x += self.step

            elif self.momentum_type == 'nesterov':

                self.x += step2
                self.step = step1 + step2

            elif self.momentum_type == 'none':

                self.step = step2
                self.x += self.step

            try:
                self.check_lagrangian_dual_optimality()
            except StopIteration:
                break

            self.iter += 1

        self.check_lagrangian_dual_conditions()

        if self.verbose:
            print('\n')

        return self
