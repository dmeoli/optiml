import matplotlib.pyplot as plt
import numpy as np

from ml.neural_network.initializers import random_uniform
from optimization.unconstrained.stochastic import StochasticOptimizer


class AdaDelta(StochasticOptimizer):

    def __init__(self, f, wrt=random_uniform, batch_size=None, eps=1e-6, max_iter=1000, step_rate=1.,
                 momentum_type='none', momentum=0.9, decay=0.95, offset=1e-4, verbose=False, plot=False):
        super().__init__(f, wrt, step_rate, momentum_type, momentum, batch_size, eps, max_iter, verbose, plot)
        if not 0 <= decay < 1:
            raise ValueError('decay has to lie in [0, 1)')
        self.decay = decay
        if not np.isscalar(offset):
            raise ValueError('offset is not a real scalar')
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset
        self.gms = 0
        self.sms = 0

    def minimize(self):

        if self.verbose and not self.iter % self.verbose:
            print('iter\tf(x)\t\t||g(x)||', end='')
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

            if self.momentum_type == 'standard':
                step_m1 = self.step
                step1 = self.momentum * step_m1
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                step1 = self.momentum * step_m1
                self.wrt -= step1

            g = self.f.jacobian(self.wrt, *args)
            self.gms = self.decay * self.gms + (1. - self.decay) * g ** 2
            delta = np.sqrt(self.sms + self.offset) / np.sqrt(self.gms + self.offset) * g

            step2 = self.step_rate * delta

            self.wrt -= step1 + step2 if self.momentum_type == 'standard' else step2
            self.step = step2 if self.momentum_type == 'none' else step1 + step2

            self.sms = self.decay * self.sms + (1. - self.decay) * self.step ** 2

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((self.wrt + self.step, self.wrt)).T
                contour_axes.quiver(p_xy[0, :-1], p_xy[1, :-1], p_xy[0, 1:] - p_xy[0, :-1], p_xy[1, 1:] - p_xy[1, :-1],
                                    scale_units='xy', angles='xy', scale=1, color='k')

            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, v, status
