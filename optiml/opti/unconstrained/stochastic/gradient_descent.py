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

            if self.is_lagrangian_dual():
                violations = self.f.AG.dot(self.x) - self.f.bh

                self.f.past_dual_x = self.f.dual_x.copy()  # backup dual_x before upgrade it

                # upgrade and clip dual_x
                self.f.dual_x += self.f.rho * violations
                self.f.dual_x[self.f.n_eq:] = np.clip(self.f.dual_x[self.f.n_eq:], a_min=0, a_max=None)

                if self.dgap <= self.tol and (np.linalg.norm(self.f.dual_x - self.f.past_dual_x) +
                                              np.linalg.norm(self.x - self.past_x) <= self.tol):
                    self.status = 'optimal'
                    break

            self.iter += 1

        if self.is_lagrangian_dual():
            assert all(self.f.dual_x[self.f.n_eq:] >= 0)  # Lagrange multipliers

        if self.verbose:
            print('\n')

        return self
