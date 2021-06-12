import numpy as np

from . import StochasticOptimizer


class AdaDelta(StochasticOptimizer):

    def __init__(self,
                 f,
                 x=None,
                 batch_size=None,
                 eps=1e-6,
                 epochs=1000,
                 step_size=1.,
                 decay=0.95,
                 offset=1e-4,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        super(AdaDelta, self).__init__(f=f,
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
        if not 0 <= decay < 1:
            raise ValueError('decay has to lie in [0, 1)')
        self.decay = decay
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset
        self.gms = np.zeros_like(self.x)
        self.sms = np.zeros_like(self.x)

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

            self.g_x = self.f.jacobian(self.x, *batch)

            # compute search direction
            d = -self.g_x

            self.gms = self.decay * self.gms + (1. - self.decay) * self.g_x ** 2
            step = self.step_size * d * (np.sqrt(self.sms + self.offset) / np.sqrt(self.gms + self.offset))

            self.x += step

            if self.is_lagrangian_dual():
                constraints = self.f.AG.dot(self.x) - self.f.bh

                self.f.past_dual_x = self.f.dual_x.copy()  # backup dual_x before upgrade it

                # upgrade and clip dual_x
                self.f.dual_x += self.f.rho * constraints
                self.f.dual_x[self.f.n_eq:] = np.clip(self.f.dual_x[self.f.n_eq:], a_min=0, a_max=None)

                if (np.linalg.norm(self.f.dual_x - self.f.past_dual_x) +
                        np.linalg.norm(self.x - self.past_x) <= self.tol):
                    self.status = 'optimal'
                    break

            self.sms = self.decay * self.sms + (1. - self.decay) * step ** 2

            self.iter += 1

        if self.is_lagrangian_dual():
            assert all(self.f.dual_x[self.f.n_eq:] >= 0)  # Lagrange multipliers

        if self.verbose:
            print('\n')

        return self
