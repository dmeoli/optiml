import numpy as np

from . import StochasticMomentumOptimizer


class StochasticGradientDescent(StochasticMomentumOptimizer):

    def __init__(self,
                 f,
                 x=None,
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

            if self.momentum_type == 'nesterov':
                step_m1 = self.step
                jump = self.momentum * step_m1
                self.x += jump

            self.g_x = self.f.jacobian(self.x, *batch)

            # compute search direction
            d = -self.g_x

            if self.is_lagrangian_dual():
                # project the direction over the active constraints
                d[np.logical_and(self.x <= 1e-12, d < 0, self.f.constrained_idx.copy())] = 0

                # first, compute the maximum feasible step size max_t such that:
                #
                #   0 <= lambda[i] + max_t * d[i]   for all i
                #     -lambda[i] <= max_t * d[i]
                #     -lambda[i] / d[i] <= max_t

                idx = d[self.f.constrained_idx] < 0  # negative gradient entries
                if any(idx):
                    max_t = min(self.step_size, min(-self.x[self.f.constrained_idx][idx] /
                                                    d[self.f.constrained_idx][idx]))
                    self.step_size = max_t

            if self.momentum_type == 'polyak':

                step_m1 = self.step
                self.step = self.step_size * d + self.momentum * step_m1
                self.x += self.step

            elif self.momentum_type == 'nesterov':

                correction = self.step_size * d
                self.x += correction
                self.step = jump + correction

            elif self.momentum_type == 'none':

                self.step = self.step_size * d
                self.x += self.step

            self.iter += 1

        if self.verbose:
            print('\n')

        if self.is_lagrangian_dual():
            assert all(self.x[self.f.constrained_idx] >= 0)  # Lagrange multipliers

        return self
