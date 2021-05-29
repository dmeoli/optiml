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

            if self.is_lagrangian_dual():
                # project the direction over the active constraints
                d[np.logical_and(self.x <= 1e-12, d < 0, self.f.constrained_idx.copy())] = 0

                # first, compute the maximum feasible step size max_t such that:
                #
                #   0 <= lambda[i] + max_t * d[i] * (sqrt(sms[i] + offset) / sqrt(gsm[i] + offset))   for all i
                #     -lambda[i] <= max_t * d[i] * (sqrt(sms[i] + offset) / sqrt(gsm[i] + offset))
                #     -lambda[i] / (d[i] / (sqrt(sms[i] + offset) * sqrt(gsm[i] + offset)) <= max_t

                idx = d[self.f.constrained_idx] < 0  # negative gradient entries
                if any(idx):
                    max_t = min(-self.x[self.f.constrained_idx][idx] /
                                (d[self.f.constrained_idx][idx] /
                                 (np.sqrt(self.sms[self.f.constrained_idx][idx] + self.offset) *
                                  np.sqrt(self.gms[self.f.constrained_idx][idx] + self.offset))))
                    self.step_size = max_t

            self.gms = self.decay * self.gms + (1. - self.decay) * self.g_x ** 2
            step = self.step_size * d * (np.sqrt(self.sms + self.offset) / np.sqrt(self.gms + self.offset))

            self.x += step

            self.sms = self.decay * self.sms + (1. - self.decay) * step ** 2

            self.iter += 1

        if self.verbose:
            print('\n')

        if self.is_lagrangian_dual():
            assert all(self.x[self.f.constrained_idx] >= 0)  # Lagrange multipliers

        return self
