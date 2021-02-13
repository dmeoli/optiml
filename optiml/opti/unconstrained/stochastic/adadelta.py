import numpy as np

from . import StochasticOptimizer


class AdaDelta(StochasticOptimizer):

    def __init__(self,
                 f,
                 x=np.random.uniform,
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
        self.gms = 0
        self.sms = 0

    def minimize(self):

        self._print_header()

        for batch in self.batches:
            self.f_x, self.g_x = self.f.function(self.x, *batch), self.f.jacobian(self.x, *batch)

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

            self.gms = self.decay * self.gms + (1. - self.decay) * self.g_x ** 2
            delta = np.sqrt(self.sms + self.offset) / np.sqrt(self.gms + self.offset) * self.g_x

            step = self.step_size * delta

            self.x -= step

            self.sms = self.decay * self.sms + (1. - self.decay) * step ** 2

            self.iter += 1

        if self.verbose:
            print('\n')

        return self
