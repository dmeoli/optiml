import numpy as np

from . import StochasticMomentumOptimizer


class StochasticGradientDescent(StochasticMomentumOptimizer):

    def __init__(self,
                 f,
                 x=np.random.uniform,
                 batch_size=None,
                 eps=1e-6,
                 epochs=1000,
                 step_size=0.01,
                 momentum_type='none',
                 momentum=0.9,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        super().__init__(f=f,
                         x=x,
                         step_size=step_size,
                         momentum_type=momentum_type,
                         momentum=momentum,
                         batch_size=batch_size,
                         eps=eps,
                         epochs=epochs,
                         callback=callback,
                         callback_args=callback_args,
                         shuffle=shuffle,
                         random_state=random_state,
                         verbose=verbose)

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

            if self.momentum_type == 'standard':

                self.g_x = self.f.jacobian(self.x, *batch)
                step_m1 = self.step

                # compute search direction
                d = -self.g_x

                if hasattr(self.f, 'primal'):
                    # project the direction over the active constraints
                    d[np.logical_and(self.x <= 1e-12, d < 0)] = 0

                self.step = self.step_size * d + self.momentum * step_m1
                self.x += self.step

            elif self.momentum_type == 'nesterov':

                step_m1 = self.step
                big_jump = self.momentum * step_m1
                self.x += big_jump
                self.g_x = self.f.jacobian(self.x, *batch)

                # compute search direction
                d = -self.g_x

                if hasattr(self.f, 'primal'):
                    # project the direction over the active constraints
                    d[np.logical_and(self.x <= 1e-12, d < 0)] = 0

                correction = self.step_size * d
                self.x += correction
                self.step = big_jump + correction

            elif self.momentum_type == 'none':

                self.g_x = self.f.jacobian(self.x, *batch)

                # compute search direction
                d = -self.g_x

                if hasattr(self.f, 'primal'):
                    # project the direction over the active constraints
                    d[np.logical_and(self.x <= 1e-12, d < 0)] = 0

                self.step = self.step_size * d
                self.x += self.step

            self.iter += 1

        if self.verbose:
            print('\n')

        # if hasattr(self.f, 'primal'):
        #     assert all(self.x >= 0)  # Lagrange multipliers

        return self
