import warnings

import numpy as np

from . import StochasticMomentumOptimizer


class AMSGrad(StochasticMomentumOptimizer):
    """
    AMSGrad for the minimization of the provided function f.

    It is a variant of Adam that, instead of the bias-corrected second moment,
    uses the element-wise maximum of all the past second raw moment estimates to
    scale the step. This keeps the effective learning rate non-increasing and
    fixes a convergence issue of Adam.

    References
    ----------
    .. [1] Reddi, S. J., Kale, S. & Kumar, S. (2018). On the Convergence of Adam and Beyond.
    """

    def __init__(self,
                 f,
                 x=None,
                 batch_size=None,
                 eps=1e-6,
                 tol=1e-8,
                 epochs=1000,
                 step_size=0.001,
                 momentum_type='none',
                 momentum=0.9,
                 beta1=0.9,
                 beta2=0.999,
                 offset=1e-8,
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
        :param step_size:     (real scalar > 0, callable or iterable, optional, default value 0.001): the
                              learning rate, i.e., the base size of the step taken along the search direction.
        :param momentum_type: (string in {'none', 'polyak', 'nesterov'}, optional, default value 'none'):
                              the kind of (outer) momentum applied on top of the AMSGrad step.
        :param momentum:      (real scalar in [0, 1) or iterable, optional, default value 0.9): the (outer)
                              momentum factor, i.e., the fraction of the previous step retained.
        :param beta1:         (real scalar in [0, 1), optional, default value 0.9): the exponential decay
                              rate for the first moment (mean) estimate of the gradient.
        :param beta2:         (real scalar in [0, 1), optional, default value 0.999): the exponential decay
                              rate for the second raw moment (uncentered variance) estimate of the gradient.
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
        super(AMSGrad, self).__init__(f=f,
                                      x=x,
                                      step_size=step_size,
                                      momentum_type=momentum_type,
                                      momentum=momentum,
                                      batch_size=batch_size,
                                      eps=eps,
                                      tol=tol,
                                      epochs=epochs,
                                      callback=callback,
                                      callback_args=callback_args,
                                      shuffle=shuffle,
                                      random_state=random_state,
                                      verbose=verbose)
        if not 0 <= beta1 < 1:
            raise ValueError('beta1 has to lie in [0, 1)')
        self.beta1 = beta1
        self.est_mom1 = 0  # initialize 1st moment vector
        if not 0 <= beta2 < 1:
            raise ValueError('beta2 has to lie in [0, 1)')
        self.beta2 = beta2
        self.est_mom2 = 0  # initialize 2nd moment vector
        if not self.beta1 < np.sqrt(self.beta2):
            warnings.warn('constraint from convergence analysis for adam not satisfied')
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset

    def minimize(self):

        self._print_header()

        est_mom2_crt = 0.

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

            est_mom1_m1 = self.est_mom1
            est_mom2_m1 = self.est_mom2

            # update biased 1st moment estimate
            self.est_mom1 = self.beta1 * est_mom1_m1 + (1. - self.beta1) * d
            # update biased 2nd raw moment estimate
            self.est_mom2 = self.beta2 * est_mom2_m1 + (1. - self.beta2) * self.g_x ** 2

            est_mom2_crt = np.maximum(self.est_mom2, est_mom2_crt)

            step2 = next(self.step_size(*batch)) * self.est_mom1 / (np.sqrt(est_mom2_crt) + self.offset)

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
