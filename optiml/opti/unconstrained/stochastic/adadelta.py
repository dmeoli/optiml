import numpy as np

from . import StochasticOptimizer
from .schedules import constant


class AdaDelta(StochasticOptimizer):

    def __init__(self,
                 f,
                 x,
                 batch_size=None,
                 eps=1e-6,
                 epochs=1000,
                 step_size=1.,
                 momentum_type='none',
                 momentum=0.9,
                 decay=0.95,
                 offset=1e-4,
                 step_size_schedule=constant,
                 momentum_schedule=constant,
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
                         step_size_schedule=step_size_schedule,
                         momentum_schedule=momentum_schedule,
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

        if self.verbose:
            print('epoch\titer\t cost\t', end='')
            if self.f.f_star() < np.inf:
                print('\t gap\t\t rate', end='')
                prev_v = np.inf

        for batch in self.batches:
            self.f_x, self.g_x = self.f.function(self.x, *batch), self.f.jacobian(self.x, *batch)

            if self.is_batch_end():

                if self.is_verbose():
                    print('\n{:4d}\t{:4d}\t{: 1.4e}'.format(self.epoch, self.iter, self.f_x), end='')
                    if self.f.f_star() < np.inf:
                        print('\t{: 1.4e}'.format(self.f_x - self.f.f_star()), end='')
                        if prev_v < np.inf:
                            print('\t{: 1.4e}'.format((self.f_x - self.f.f_star()) /
                                                      (prev_v - self.f.f_star())), end='')
                        else:
                            print('\t\t', end='')
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
                step1 = self.momentum * step_m1
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                step1 = self.momentum * step_m1
                self.x -= step1

            self.g_x = self.f.jacobian(self.x, *batch)
            self.gms = self.decay * self.gms + (1. - self.decay) * self.g_x ** 2
            delta = np.sqrt(self.sms + self.offset) / np.sqrt(self.gms + self.offset) * self.g_x

            step2 = self.step_size * delta

            if self.momentum_type == 'standard':
                self.x -= step1 + step2
            else:
                self.x -= step2

            if self.momentum_type != 'none':
                self.step = step1 + step2
            else:
                self.step = step2

            self.sms = self.decay * self.sms + (1. - self.decay) * self.step ** 2

            self.iter += 1

        if self.verbose:
            print('\n')

        return self
