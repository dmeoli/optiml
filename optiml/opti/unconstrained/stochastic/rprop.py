import numpy as np

from . import StochasticOptimizer


class RProp(StochasticOptimizer):

    def __init__(self,
                 f,
                 x=None,
                 batch_size=None,
                 eps=1e-6,
                 epochs=1000,
                 step_size=0.001,
                 min_step=1e-6,
                 step_shrink=0.5,
                 step_grow=1.2,
                 max_step=1,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        super().__init__(f=f,
                         x=x,
                         step_size=step_size,
                         batch_size=batch_size,
                         eps=eps,
                         epochs=epochs,
                         callback=callback,
                         callback_args=callback_args,
                         shuffle=shuffle,
                         random_state=random_state,
                         verbose=verbose)
        self.min_step = min_step
        self.step_shrink = step_shrink
        self.step_grow = step_grow
        self.max_step = max_step

    def minimize(self):

        self._print_header()

        g_x_m1 = np.zeros_like(self.x)
        changes = np.zeros_like(self.x)

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

            self.g_x = self.f.jacobian(self.x, *batch)

            # compute search direction
            d = -self.g_x

            if self.is_lagrangian_dual():
                # project the direction over the active constraints
                d[np.logical_and(self.x <= 1e-12, d < 0)] = 0

            grad_prod = g_x_m1 * d

            changes[grad_prod > 0] *= self.step_grow
            changes[grad_prod < 0] *= self.step_shrink
            changes = np.clip(changes, self.min_step, self.max_step)

            step = changes * np.sign(self.g_x)

            self.x += step

            self.iter += 1

            g_x_m1 = self.g_x

        if self.verbose:
            print('\n')

        # if self.is_lagrangian_dual():
        #     assert all(self.x >= 0)  # Lagrange multipliers

        return self
