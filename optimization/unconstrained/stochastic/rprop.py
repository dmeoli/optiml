import matplotlib.pyplot as plt
import numpy as np

from ml.neural_network.initializers import random_uniform
from optimization.unconstrained.stochastic.stochastic_optimizer import StochasticOptimizer


class RProp(StochasticOptimizer):

    def __init__(self, f, wrt=random_uniform, batch_size=None, eps=1e-6, max_iter=1000,
                 step_rate=0.001, min_step=1e-6, step_shrink=0.5, step_grow=1.2, max_step=1,
                 momentum_type='none', momentum=0.9, verbose=False, plot=False):
        super().__init__(f, wrt, step_rate, momentum_type, momentum, batch_size, eps, max_iter, verbose, plot)
        self.min_step = min_step
        self.step_shrink = step_shrink
        self.step_grow = step_grow
        self.max_step = max_step
        self.jacobian = np.zeros_like(self.wrt)
        self.changes = np.zeros_like(self.wrt)

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

            g_m1 = self.jacobian

            self.jacobian = self.f.jacobian(self.wrt, *args)
            grad_prod = g_m1 * self.jacobian

            self.changes[grad_prod > 0] *= self.step_grow
            self.changes[grad_prod < 0] *= self.step_shrink
            self.changes = np.clip(self.changes, self.min_step, self.max_step)

            step2 = self.changes * np.sign(self.jacobian)

            self.wrt -= step1 + step2 if self.momentum_type == 'standard' else step2
            self.step = step2 if self.momentum_type == 'none' else step1 + step2

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
