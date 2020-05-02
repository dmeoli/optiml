import warnings

import matplotlib.pyplot as plt
import numpy as np

from ml.neural_network.initializers import random_uniform
from optimization.unconstrained.stochastic.stochastic_optimizer import StochasticOptimizer


class AdaMax(StochasticOptimizer):

    def __init__(self, f, x=random_uniform, batch_size=None, eps=1e-6, epochs=1000, step_size=0.002,
                 momentum_type='none', momentum=0.9, beta1=0.9, beta2=0.999, offset=1e-8, callback=None,
                 callback_args=(), verbose=False, plot=False):
        super().__init__(f, x, step_size, momentum_type, momentum, batch_size,
                         eps, epochs, callback, callback_args, verbose, plot)
        if not 0 <= beta1 < 1:
            raise ValueError('beta1 has to lie in [0, 1)')
        self.beta1 = beta1
        self.est_mom1 = 0  # initialize 1st moment vector
        if not 0 <= beta2 < 1:
            raise ValueError('beta2 has to lie in [0, 1)')
        self.beta2 = beta2
        self.est_mom2 = 0  # initialize the exponentially weighted infinity norm
        if not self.beta1 < np.sqrt(self.beta2):
            warnings.warn('constraint from convergence analysis for adam not satisfied')
        if not np.isscalar(offset):
            raise ValueError('offset is not a real scalar')
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset

    def minimize(self):

        if self.verbose and not self.iter % self.verbose:
            print('iter\t\tf(x)\t\t||g(x)||', end='')
            if self.f.f_star() < np.inf:
                print('\tf(x) - f*\trate', end='')
                prev_v = np.inf

        if self.plot:
            fig = self.f.plot()

        for args in self.args:
            self.f_x, g = self.f.function(self.x, *args), self.f.jacobian(self.x, *args)
            ng = np.linalg.norm(g)

            if self.verbose and not self.iter % self.verbose:
                print('\n{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, self.f_x, ng), end='')
                if self.f.f_star() < np.inf:
                    print('\t{:1.4e}'.format(self.f_x - self.f.f_star()), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((self.f_x - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    prev_v = self.f_x

            # stopping criteria
            if ng <= self.eps:
                status = 'optimal'
                break

            if self.iter >= self.max_iter:
                status = 'stopped'
                break

            t = self.iter + 1

            if self.momentum_type == 'standard':
                step_m1 = self.step
                step1 = self.momentum * step_m1
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                step1 = self.momentum * step_m1
                self.x -= step1

            est_mom1_m1 = self.est_mom1
            est_mom2_m1 = self.est_mom2

            g = self.f.jacobian(self.x, *args)
            self.est_mom1 = self.beta1 * est_mom1_m1 + (1. - self.beta1) * g  # update biased 1st moment estimate
            # update the exponentially weighted infinity norm
            self.est_mom2 = np.maximum(self.beta2 * est_mom2_m1, np.abs(g))

            est_mom1_crt = self.est_mom1 / (1. - self.beta1 ** t)  # compute bias-corrected 1st moment estimate

            step2 = self.step_size * est_mom1_crt / (self.est_mom2 + self.offset)

            if self.momentum_type == 'standard':
                self.x -= step1 + step2
            else:
                self.x -= step2

            if self.momentum_type in ('standard', 'nesterov'):
                self.step = step1 + step2
            else:
                self.step = step2

            # plot the trajectory
            if self.plot:
                super().plot_step(fig, self.x + self.step, self.x)

            self.iter += 1

            self.callback()

        if self.verbose:
            print()
        if self.plot:
            plt.show()
        return self.x, self.f_x, status
