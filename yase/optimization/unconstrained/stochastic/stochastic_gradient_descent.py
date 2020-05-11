import numpy as np

from . import StochasticOptimizer
from .schedules import constant


class StochasticGradientDescent(StochasticOptimizer):

    def __init__(self, f, x, batch_size=None, eps=1e-6, epochs=1000, step_size=0.01, momentum_type='none',
                 momentum=0.9, step_size_schedule=constant, momentum_schedule=constant, callback=None,
                 callback_args=(), shuffle=True, random_state=None, verbose=False):
        super().__init__(f, x, step_size, momentum_type, momentum, batch_size, eps, epochs,
                         callback, callback_args, shuffle, random_state, verbose)
        self.step_size = step_size_schedule(self.step_size)
        self.momentum = momentum_schedule(self.momentum)

    def minimize(self):

        if self.verbose:
            print('epoch\tf(x)\t', end='')
            if self.f.f_star() < np.inf:
                print('\tf(x) - f*\trate', end='')
                prev_v = np.inf

        for batch in self.batches:
            self.f_x, self.g_x = self.f.function(self.x, *batch), self.f.jacobian(self.x, *batch)

            if self.is_batch_end():

                if self.verbose and not self.epoch % self.verbose:
                    print('\n{:4d}\t{:1.4e}'.format(self.epoch, self.f_x), end='')
                    if self.f.f_star() < np.inf:
                        print('\t{:1.4e}'.format(self.f_x - self.f.f_star()), end='')
                        if prev_v < np.inf:
                            print('\t{:1.4e}'.format((self.f_x - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                        prev_v = self.f_x

                self.callback(batch)
                self.epoch += 1

            if self.epoch >= self.epochs:
                self.status = 'stopped'
                break

            if self.momentum_type == 'standard':
                step_m1 = self.step
                self.step = next(self.step_size) * -self.g_x + next(self.momentum) * step_m1
                self.x += self.step
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                big_jump = next(self.momentum) * step_m1
                self.x += big_jump
                self.g_x = self.f.jacobian(self.x, *batch)
                correction = next(self.step_size) * -self.g_x
                self.x += correction
                self.step = big_jump + correction
            elif self.momentum_type == 'none':
                self.step = next(self.step_size) * -self.g_x
                self.x += self.step

            self.iter += 1

        if self.verbose:
            print('\n')

        return self
