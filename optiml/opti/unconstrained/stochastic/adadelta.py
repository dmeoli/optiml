import numpy as np

from . import StochasticOptimizer


class AdaDelta(StochasticOptimizer):
    """
    AdaDelta for the minimization of the provided function f.

    It is an extension of AdaGrad that replaces the ever-growing sum of squared
    gradients with an exponentially decaying average and, by also tracking an
    exponentially decaying average of the squared updates, scales the step by the
    ratio of these two running averages, removing the need for a manually tuned
    global learning rate.

    References
    ----------
    .. [1] Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method.
    """

    def __init__(self,
                 f,
                 x=None,
                 batch_size=None,
                 eps=1e-6,
                 tol=1e-8,
                 epochs=1000,
                 step_size=1.,
                 decay=0.9,
                 offset=1e-6,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        """

        :param f:             the objective function.
        :param x:             ([n x 1] real column vector): the point where to start the algorithm from.
        :param batch_size:    (integer scalar or None, optional, default value None): the size of the mini
                              batches used to estimate the gradient; if None the full sample is used.
        :param eps:           (real scalar, optional, default value 1e-6): the accuracy in the stopping
                              criterion: the algorithm is stopped when the norm of the gradient is less
                              than or equal to eps.
        :param tol:           (real scalar, optional, default value 1e-8): the tolerance used in the
                              optimality conditions of the Lagrangian dual (when f is a Lagrangian dual).
        :param epochs:        (integer scalar, optional, default value 1000): the maximum number of epochs
                              before the algorithm is stopped.
        :param step_size:     (real scalar > 0, callable or iterable, optional, default value 1.): the
                              learning rate, i.e., the base size of the step taken along the negative gradient.
        :param decay:         (real scalar in [0, 1), optional, default value 0.9): the exponential decay
                              rate of the running averages of the squared gradients and of the squared updates.
        :param offset:        (real scalar > 0, optional, default value 1e-6): a small constant added to the
                              running averages to avoid division by zero and improve numerical stability.
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
        super(AdaDelta, self).__init__(f=f,
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
        if not 0 <= decay < 1:
            raise ValueError('decay has to lie in [0, 1)')
        self.decay = decay
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset
        self.gms = np.zeros_like(self.x)
        self.sms = np.zeros_like(self.x)

    def minimize(self):

        self._print_header()

        for batch in self.batches:

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

            self.gms = self.decay * self.gms + (1. - self.decay) * self.g_x ** 2
            self.step = (next(self.step_size(*batch)) * d * (np.sqrt(self.sms + self.offset) /
                                                             np.sqrt(self.gms + self.offset)))

            self.x += self.step

            try:
                self.check_lagrangian_dual_optimality()
            except StopIteration:
                break

            self.sms = self.decay * self.sms + (1. - self.decay) * self.step ** 2

            self.iter += 1

        self.check_lagrangian_dual_conditions()

        if self.verbose:
            print('\n')

        return self
