import warnings

import numpy as np

from . import StochasticMomentumOptimizer


class AdaMax(StochasticMomentumOptimizer):

    def __init__(self,
                 f,
                 x=np.random.uniform,
                 batch_size=None,
                 eps=1e-6,
                 epochs=1000,
                 step_size=0.002,
                 momentum_type='none',
                 momentum=0.9,
                 beta1=0.9,
                 beta2=0.999,
                 offset=1e-8,
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
        if not 0 <= beta1 < 1:
            raise ValueError('beta1 has to lie in [0, 1)')
        self.beta1 = beta1
        self.est_mom1 = 0  # initialize 1st moment vector
        if not 0 <= beta2 < 1:
            raise ValueError('beta2 has to lie in [0, 1)')
        self.beta2 = beta2
        self.est_mom2 = 0  # initialize the exponentially weighted infinity norm
        if not self.beta1 < np.sqrt(self.beta2):
            warnings.warn('constraint from convergence analysis for adam not satisfied')
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset

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
                self.epoch += 1

            if self.epoch >= self.epochs:
                self.status = 'stopped'
                break

            t = self.iter + 1

            if self.momentum_type == 'standard':
                step_m1 = self.step
                step1 = self.momentum * step_m1
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                step1 = self.momentum * step_m1
                self.x -= step1

            est_mom1_m1 = self.est_mom1
            est_mom2_m1 = self.est_mom2

            self.g_x = self.f.jacobian(self.x, *batch)
            self.est_mom1 = self.beta1 * est_mom1_m1 + (1. - self.beta1) * self.g_x  # update biased 1st moment estimate
            # update the exponentially weighted infinity norm
            self.est_mom2 = np.maximum(self.beta2 * est_mom2_m1, np.abs(self.g_x))

            est_mom1_crt = self.est_mom1 / (1. - self.beta1 ** t)  # compute bias-corrected 1st moment estimate

            step2 = self.step_size * est_mom1_crt / (self.est_mom2 + self.offset)

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
