import numpy as np

from ml.neural_network.initializers import random_uniform
from optimization.unconstrained.stochastic.stochastic_optimizer import StochasticOptimizer


class AdaDelta(StochasticOptimizer):

    def __init__(self, f, x=random_uniform, batch_size=None, eps=1e-6, epochs=1000, step_size=1.,
                 momentum_type='none', momentum=0.9, decay=0.95, offset=1e-4, callback=None,
                 callback_args=(), shuffle=True, random_state=None, verbose=False):
        super().__init__(f, x, step_size, momentum_type, momentum, batch_size, eps, epochs,
                         callback, callback_args, shuffle, random_state, verbose)
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
            print('epoch\tf(x)', end='')
            if self.f.f_star() < np.inf:
                print('\tf(x) - f*\trate', end='')
                prev_v = np.inf

        for args in self.args:
            self.f_x, g = self.f.function(self.x, *args), self.f.jacobian(self.x, *args)

            self.callback(args)

            if self.verbose and not self.iter % self.verbose:
                print('\n{:4d}\t{:1.4e}'.format(self.iter, self.f_x), end='')
                if self.f.f_star() < np.inf:
                    print('\t{:1.4e}'.format(self.f_x - self.f.f_star()), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((self.f_x - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    prev_v = self.f_x

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
            self.gms = self.decay * self.gms + (1. - self.decay) * g ** 2
            delta = np.sqrt(self.sms + self.offset) / np.sqrt(self.gms + self.offset) * g

            step2 = self.step_size * delta

            if self.momentum_type == 'standard':
                self.x -= step1 + step2
            else:
                self.x -= step2

            if self.momentum_type in ('standard', 'nesterov'):
                self.step = step1 + step2
            else:
                self.step = step2

            self.sms = self.decay * self.sms + (1. - self.decay) * self.step ** 2

            self.iter += 1

        if self.verbose:
            print('\n')
        return self.x, self.f_x, status
