import numpy as np

from . import StochasticMomentumOptimizer


class RMSProp(StochasticMomentumOptimizer):

    def __init__(self,
                 f,
                 x=None,
                 step_size=0.001,
                 momentum_type='none',
                 momentum=0.9,
                 batch_size=None,
                 eps=1e-6,
                 epochs=1000,
                 decay=0.9,
                 offset=1e-8,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        super().__init__(f=f,
                         x=x,
                         step_size=step_size,
                         momentum=momentum,
                         momentum_type=momentum_type,
                         batch_size=batch_size,
                         eps=eps,
                         epochs=epochs,
                         callback=callback,
                         callback_args=callback_args,
                         shuffle=shuffle,
                         random_state=random_state,
                         verbose=verbose)
        if not 0 <= decay < 1:
            raise ValueError('decay has to lie in [0, 1)')
        self.decay = decay
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset
        self.moving_mean_squared = np.ones_like(self.x)

    def minimize(self):

        self._print_header()

        for batch in self.batches:

            self.f_x = self.f.function(self.x, *batch)

            self._print_info()

            try:
                self.callback(batch)
            except StopIteration:
                break

            if self.is_batch_end():
                self.epoch += 1

            if self.epoch >= self.epochs:
                self.status = 'stopped'
                break

            if self.momentum_type == 'nesterov':
                step_m1 = self.step
                step1 = self.momentum * step_m1
                self.x += step1

            self.g_x = self.f.jacobian(self.x, *batch)

            # compute search direction
            d = -self.g_x

            self.moving_mean_squared = self.decay * self.moving_mean_squared + (1. - self.decay) * self.g_x ** 2
            step2 = self.step_size * d / np.sqrt(self.moving_mean_squared + self.offset)

            if self.momentum_type == 'standard':

                step_m1 = self.step
                self.step = self.momentum * step_m1 + step2
                self.x += self.step

            elif self.momentum_type == 'nesterov':

                self.x += step2
                self.step = step1 + step2

            elif self.momentum_type == 'none':

                self.step = step2
                self.x += self.step

            self.iter += 1

        if self.verbose:
            print('\n')

        return self
