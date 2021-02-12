import numpy as np

from . import StochasticOptimizer


class AdaGrad(StochasticOptimizer):

    def __init__(self,
                 f,
                 x=np.random.uniform,
                 batch_size=None,
                 eps=1e-6,
                 epochs=1000,
                 step_size=0.01,
                 momentum_type='none',
                 momentum=0.9,
                 offset=1e-4,
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
        if not offset > 0:
            raise ValueError('offset must be > 0')
        self.offset = offset
        self.gms = 0

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

            self.gms += self.g_x ** 2
            step = self.step_size * self.g_x / np.sqrt(self.gms + self.offset)

            self.x -= step

            self.iter += 1

        if self.verbose:
            print('\n')

        return self
