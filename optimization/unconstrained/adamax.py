import warnings

import matplotlib.pyplot as plt
import numpy as np

from ml.initializers import random_uniform
from optimization.optimizer import Optimizer


class AdaMax(Optimizer):

    def __init__(self, f, wrt=random_uniform, batch_size=None, eps=1e-6, max_iter=1000, step_rate=0.002,
                 momentum_type='none', momentum=0.9, beta1=0.9, beta2=0.999, offset=1e-8, verbose=False, plot=False):
        super().__init__(f, wrt, batch_size, eps, max_iter, verbose, plot)
        if not np.isscalar(step_rate):
            raise ValueError('step_rate is not a real scalar')
        if not step_rate > 0:
            raise ValueError('step_rate must be > 0')
        self.step_rate = step_rate
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
        if not np.isscalar(momentum):
            raise ValueError('momentum is not a real scalar')
        if not momentum > 0:
            raise ValueError('momentum must be > 0')
        self.momentum = momentum
        if momentum_type not in ('standard', 'nesterov', 'none'):
            raise ValueError(f'unknown momentum type {momentum_type}')
        self.momentum_type = momentum_type
        if not np.isscalar(offset):
            raise ValueError('offset is not a real scalar')
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset
        self.step = 0

    def minimize(self):

        if self.verbose and not self.iter % self.verbose:
            print('iter\t\tf(x)\t\t||g(x)||', end='')
            if self.f.f_star() < np.inf:
                print('\tf(x) - f*\trate', end='')
                prev_v = np.inf

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        for args in self.args:
            v, g = self.f.function(self.wrt, *args), self.f.jacobian(self.wrt, *args)
            ng = np.linalg.norm(g)

            if self.verbose and not self.iter % self.verbose:
                print('\n{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, v, ng), end='')
                if self.f.f_star() < np.inf:
                    print('\t{:1.4e}'.format(v - self.f.f_star()), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((v - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    prev_v = v

            # stopping criteria
            if ng <= self.eps:
                status = 'optimal'
                break

            if self.iter >= self.max_iter:
                status = 'stopped'
                break

            t = self.iter

            if self.momentum_type == 'standard':
                step_m1 = self.step
                step1 = self.momentum * step_m1
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                step1 = self.momentum * step_m1
                self.wrt -= step1

            est_mom1_m1 = self.est_mom1
            est_mom2_m1 = self.est_mom2

            g = self.f.jacobian(self.wrt, *args)
            self.est_mom1 = self.beta1 * est_mom1_m1 + (1. - self.beta1) * g  # update biased 1st moment estimate
            # update the exponentially weighted infinity norm
            self.est_mom2 = np.maximum(self.beta2 * est_mom2_m1, np.abs(g))

            est_mom1_crt = self.est_mom1 / (1. - self.beta1 ** t)  # compute bias-corrected 1st moment estimate

            step2 = self.step_rate * est_mom1_crt / (self.est_mom2 + self.offset)

            self.wrt -= step1 + step2 if self.momentum_type == 'standard' else step2
            self.step = step2 if self.momentum_type == 'none' else step1 + step2

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((self.wrt + self.step, self.wrt)).T
                contour_axes.quiver(p_xy[0, :-1], p_xy[1, :-1], p_xy[0, 1:] - p_xy[0, :-1], p_xy[1, 1:] - p_xy[1, :-1],
                                    scale_units='xy', angles='xy', scale=1, color='k')

            self.iter += 1

        if self.verbose and not self.iter % self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, v, status
