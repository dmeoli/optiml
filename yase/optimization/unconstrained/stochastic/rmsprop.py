import numpy as np

from . import StochasticOptimizer


class RMSProp(StochasticOptimizer):

    def __init__(self, f, x, batch_size=None, eps=1e-6, epochs=1000, step_size=0.001, decay=0.9, momentum_type='none',
                 momentum=0.9, callback=None, callback_args=(), shuffle=True, random_state=None, verbose=False):
        super().__init__(f, x, step_size, momentum_type, momentum, batch_size, eps, epochs,
                         callback, callback_args, shuffle, random_state, verbose)
        if not 0 <= decay < 1:
            raise ValueError('decay has to lie in [0, 1)')
        self.decay = decay
        self.moving_mean_squared = 1

    def minimize(self):

        if self.verbose:
            print('epoch\titer\tcost\t', end='')
            if self.f.f_star() < np.inf:
                print('\tgap\t\trate', end='')
                prev_v = np.inf

        for batch in self.batches:
            self.f_x, self.g_x = self.f.function(self.x, *batch), self.f.jacobian(self.x, *batch)

            if self.is_batch_end():

                if self.is_verbose():
                    print('\n{:4d}\t{:4d}\t{:1.4e}'.format(self.epoch, self.iter, self.f_x), end='')
                    if self.f.f_star() < np.inf:
                        print('\t{:1.4e}'.format(self.f_x - self.f.f_star()), end='')
                        if prev_v < np.inf:
                            print('\t{:1.4e}'.format((self.f_x - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                        else:
                            print('\t\t', end='')
                        prev_v = self.f_x

            try:
                self.callback(batch)
            except StopIteration:
                break

            if self.is_batch_end():
                self.epoch += 1

            if self.epoch >= self.epochs:
                self.status = 'stopped'
                break

            if self.momentum_type == 'standard':
                step_m1 = self.step
                step1 = self.momentum * step_m1
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                step1 = self.momentum * step_m1
                self.x -= step1

            self.g_x = self.f.jacobian(self.x, *batch)

            self.moving_mean_squared = self.decay * self.moving_mean_squared + (1. - self.decay) * self.g_x ** 2
            step2 = self.step_size * self.g_x / np.sqrt(self.moving_mean_squared)

            if self.momentum_type == 'standard':
                self.x -= step1 + step2
            else:
                self.x -= step2

            if self.momentum_type != 'none':
                self.step = step1 + step2
            else:
                self.step = step2

            self.iter += 1

        if self.verbose:
            print('\n')

        return self
