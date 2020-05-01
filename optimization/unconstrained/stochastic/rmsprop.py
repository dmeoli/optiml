import matplotlib.pyplot as plt
import numpy as np

from ml.neural_network.initializers import random_uniform
from optimization.unconstrained.stochastic.stochastic_optimizer import StochasticOptimizer


class RMSProp(StochasticOptimizer):

    def __init__(self, f, x=random_uniform, batch_size=None, eps=1e-6, max_iter=1000, step_size=0.001, decay=0.9,
                 momentum_type='none', momentum=0.9, callback=None, callback_args=(), verbose=False, plot=False):
        super().__init__(f, x, step_size, momentum_type, momentum, batch_size,
                         eps, max_iter, callback, callback_args, verbose, plot)
        if not 0 <= decay < 1:
            raise ValueError('decay has to lie in [0, 1)')
        self.decay = decay
        self.moving_mean_squared = 1

    def minimize(self):

        if self.verbose and not self.iter % self.verbose:
            print('iter\tf(x)\t\t||g(x)||', end='')
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

            if self.momentum_type == 'standard':
                step_m1 = self.step
                step1 = self.momentum * step_m1
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                step1 = self.momentum * step_m1
                self.x -= step1

            g = self.f.jacobian(self.x, *args)

            self.moving_mean_squared = self.decay * self.moving_mean_squared + (1. - self.decay) * g ** 2
            step2 = self.step_size * g / np.sqrt(self.moving_mean_squared)

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
