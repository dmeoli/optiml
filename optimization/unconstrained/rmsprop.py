import matplotlib.pyplot as plt
import numpy as np

from ml.initializers import random_uniform
from optimization.optimizer import Optimizer


class RMSProp(Optimizer):

    def __init__(self, f, wrt=random_uniform, batch_size=None, eps=1e-6, max_iter=1000, step_rate=0.001,
                 nesterov_momentum=False, momentum=0.9, decay=0.9, verbose=False, plot=False):
        super().__init__(f, wrt, batch_size, eps, max_iter, verbose, plot)
        if not np.isscalar(step_rate):
            raise ValueError('step_rate is not a real scalar')
        if not step_rate > 0:
            raise ValueError('step_rate must be > 0')
        self.step_rate = step_rate
        if not 0 <= decay < 1:
            raise ValueError('decay has to lie in [0, 1)')
        self.decay = decay
        if not np.isscalar(momentum):
            raise ValueError('momentum is not a real scalar')
        if not momentum > 0:
            raise ValueError('momentum must be > 0')
        self.momentum = momentum
        if not isinstance(nesterov_momentum, bool):
            raise ValueError('must be either True or False')
        self.nesterov_momentum = nesterov_momentum
        self.moving_mean_squared = 1
        self.step = 0

    def minimize(self):
        if self.verbose:
            print('iter\tf(x)\t\t||g(x)||', end='')
            if self.f.f_star() < np.inf:
                print('\tf(x) - f*\trate', end='')
                prev_v = np.inf
            print()

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        for args, kwargs in self.args:
            g = self.f.jacobian(self.wrt, *args, **kwargs)
            ng = np.linalg.norm(g)

            if self.verbose:
                v = self.f.function(self.wrt, *args, **kwargs)
                print('{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, v, ng), end='')
                if self.f.f_star() < np.inf:
                    print('\t{:1.4e}'.format(v - self.f.f_star()), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((v - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    prev_v = v
                print()

            # stopping criteria
            if ng <= self.eps:
                status = 'optimal'
                break

            if self.iter > self.max_iter:
                status = 'stopped'
                break

            step_m1 = self.step

            if self.nesterov_momentum:
                step1 = self.momentum * step_m1
                self.wrt -= step1

            g = self.f.jacobian(self.wrt, *args, **kwargs)

            self.moving_mean_squared = self.decay * self.moving_mean_squared + (1. - self.decay) * g ** 2
            step2 = self.step_rate * g / np.sqrt(self.moving_mean_squared)

            self.wrt -= step2
            self.step = step1 + step2 if self.nesterov_momentum else step2

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
        return self.wrt, status
