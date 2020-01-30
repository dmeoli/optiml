import matplotlib.pyplot as plt
import numpy as np

from optimization.optimizer import Optimizer


class Adam(Optimizer):

    def __init__(self, f, wrt=None, eps=1e-6, max_iter=1000, step_rate=0.001, beta1=0.9, beta2=0.999,
                 momentum=0.9, momentum_type='none', offset=1e-8, verbose=False, plot=False):
        super().__init__(f, wrt, eps, max_iter, verbose, plot)
        if not np.isscalar(step_rate):
            raise ValueError('step_rate is not a real scalar')
        if not step_rate > 0:
            raise ValueError('step_rate must be > 0')
        self.step_rate = step_rate
        if not 0 < beta1 <= 1:
            raise ValueError('beta1 has to lie in (0, 1]')
        self.beta1 = beta1
        self.est_mom1 = 0
        if not 0 < beta2 <= 1:
            raise ValueError('beta2 has to lie in (0, 1]')
        self.beta2 = beta2
        self.est_mom2 = 0
        if not (1 - beta1 * 2) / (1 - beta2) ** 0.5 < 1:
            raise ValueError('constraint from convergence analysis for adam not satisfied')
        if not np.isscalar(momentum):
            raise ValueError('momentum is not a real scalar')
        if not momentum > 0:
            raise ValueError('momentum must be > 0')
        self.momentum = momentum
        if momentum_type not in ('nesterov', 'standard', 'none'):
            raise ValueError('unknown momentum type {}'.format(momentum_type))
        self.momentum_type = momentum_type
        if not np.isscalar(offset):
            raise ValueError('offset is not a real scalar')
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset
        self.step = 0

    def minimize(self):
        if self.verbose:
            f_star = self.f.function(np.zeros((self.n,)))
            print('iter\tf(x)\t\t||g(x)||', end='')
            if f_star < np.inf:
                print('\tf(x) - f*\trate', end='')
                prev_v = np.inf
            print()

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        while True:
            g = self.f.jacobian(self.wrt)
            ng = np.linalg.norm(g)

            if self.verbose:
                v = self.f.function(self.wrt)
                print('{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, v, ng), end='')
                if f_star < np.inf:
                    print('\t{:1.4e}'.format(v - f_star), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((v - f_star) / (prev_v - f_star)), end='')
                    prev_v = v
                print()

            # stopping criteria
            if ng <= self.eps:
                status = 'optimal'
                break

            if self.iter > self.max_iter:
                status = 'stopped'
                break

            m = self.momentum
            dm1 = self.beta1
            dm2 = self.beta2
            o = self.offset
            t = self.iter + 1

            step_m1 = self.step
            step1 = step_m1 * m
            self.wrt = self.wrt - step1

            est_mom1_m1 = self.est_mom1
            est_mom2_m1 = self.est_mom2

            g = self.f.jacobian(self.wrt)
            self.est_mom1 = dm1 * g + (1 - dm1) * est_mom1_m1
            self.est_mom2 = dm2 * g ** 2 + (1 - dm2) * est_mom2_m1

            step_t = self.step_rate * (1 - (1 - dm2) ** t) ** 0.5 / (1 - (1 - dm1) ** t)
            step2 = step_t * self.est_mom1 / (self.est_mom2 ** 0.5 + o)

            last_wrt = self.wrt - step2
            self.step = step1 + step2

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((self.wrt, last_wrt)).T
                contour_axes.quiver(p_xy[0, :-1], p_xy[1, :-1], p_xy[0, 1:] - p_xy[0, :-1], p_xy[1, 1:] - p_xy[1, :-1],
                                    scale_units='xy', angles='xy', scale=1, color='k')

            self.wrt = last_wrt
            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, status


if __name__ == "__main__":
    import optimization.functions as tf

    print(Adam(tf.Rosenbrock(), [-1, 1], verbose=True, plot=True).minimize())
