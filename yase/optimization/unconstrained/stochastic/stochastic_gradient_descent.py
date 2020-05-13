import numpy as np

from . import StochasticOptimizer
from .schedules import constant


class StochasticGradientDescent(StochasticOptimizer):

    def __init__(self, f, x, batch_size=None, eps=1e-6, epochs=1000, step_size=0.01, momentum_type='none',
                 momentum=0.9, step_size_schedule=constant, momentum_schedule=constant, callback=None,
                 callback_args=(), shuffle=True, random_state=None, verbose=False):
        super().__init__(f, x, step_size, momentum_type, momentum, batch_size, eps, epochs,
                         callback, callback_args, shuffle, random_state, verbose)
        self.step_size_schedule = step_size_schedule(self.step_size)
        if self.momentum_type != 'none':
            self.momentum_schedule = momentum_schedule(self.momentum)

    def minimize(self):

        if self.verbose:
            print('epoch\titer\tf(x)\t', end='')
            if self.f.f_star() < np.inf:
                print('\tf(x) - f*\trate', end='')
                prev_v = np.inf

        for batch in self.batches:
            self.f_x, self.g_x = self.f.function(self.x, *batch), self.f.jacobian(self.x, *batch)

            if self.is_batch_end():

                if self.verbose and not self.epoch % self.verbose:
                    print('\n{:4d}\t{:4d}\t{:1.4e}'.format(self.epoch, self.iter, self.f_x), end='')
                    if self.f.f_star() < np.inf:
                        print('\t{:1.4e}'.format(self.f_x - self.f.f_star()), end='')
                        if prev_v < np.inf:
                            print('\t{:1.4e}'.format((self.f_x - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                        prev_v = self.f_x

            try:
                self.callback(batch)
            except StopIteration:
                break

            if self.is_batch_end():
                self.step_size = next(self.step_size_schedule)
                if self.momentum_type != 'none':
                    self.momentum = next(self.momentum_schedule)
                self.epoch += 1

            if self.epoch >= self.epochs:
                self.status = 'stopped'
                break

            if self.momentum_type == 'standard':
                step_m1 = self.step
                self.step = self.step_size * -self.g_x + self.momentum * step_m1
                self.x += self.step
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                big_jump = self.momentum * step_m1
                self.x += big_jump
                self.g_x = self.f.jacobian(self.x, *batch)
                correction = self.step_size * -self.g_x
                self.x += correction
                self.step = big_jump + correction
            elif self.momentum_type == 'none':
                self.step = self.step_size * -self.g_x
                self.x += self.step

            self.iter += 1

        if self.verbose:
            print('\n')

        return self
