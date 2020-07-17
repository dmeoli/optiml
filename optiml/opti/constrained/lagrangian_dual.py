import numpy as np

from .. import Optimizer
from ..unconstrained.line_search import LineSearchOptimizer
from ..unconstrained.stochastic import StochasticOptimizer, AdaGrad


class LagrangianDual(Optimizer):

    def __init__(self,
                 f,
                 optimizer=AdaGrad,
                 eps=1e-6,
                 step_size=0.01,
                 momentum_type='none',
                 momentum=0.9,
                 batch_size=None,
                 max_iter=1000,
                 max_f_eval=1000,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        super().__init__(f=f,
                         x=np.zeros(f.ndim),
                         eps=eps,
                         max_iter=max_iter,
                         callback=callback,
                         callback_args=callback_args,
                         verbose=verbose)
        if self.f.primal.ndim == 2:
            self.x0_history = []
            self.x1_history = []
            self.f_x_history = []
        self.optimizer = optimizer
        self.step_size = step_size
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_f_eval = max_f_eval
        self.shuffle = shuffle
        self.random_state = random_state

    def _print_dual_info(self, opt):
        primal_f_x = -self.f.primal.function(self.f.last_x)
        gap = (primal_f_x - self.f_x) / max(abs(self.f_x), 1)

        if ((isinstance(opt, LineSearchOptimizer) and opt.is_verbose()) or
            (isinstance(opt, StochasticOptimizer) and opt.is_batch_end())) and self.is_verbose():
            print('\tub: {: 1.4e}'.format(self.f_x), end='')
            print(' - pcost: {: 1.4e}'.format(primal_f_x), end='')
            print(' - gap: {: 1.4e}'.format(gap), end='')

        self.callback()

        self.x, self.f_x = self.f.last_x, primal_f_x

        if gap <= self.eps:
            self.status = 'optimal'
            raise StopIteration

        self.iter += 1

    def minimize(self):

        self.f_x = self.f.function(self.x)

        if issubclass(self.optimizer, LineSearchOptimizer):

            self.optimizer = self.optimizer(f=self.f,
                                            x=self.x,
                                            max_iter=self.max_iter,
                                            max_f_eval=self.max_f_eval,
                                            callback=self._print_dual_info,
                                            verbose=self.verbose).minimize()

        elif issubclass(self.optimizer, StochasticOptimizer):

            self.optimizer = self.optimizer(f=self.f,
                                            x=self.x,
                                            step_size=self.step_size,
                                            epochs=self.max_iter,
                                            batch_size=self.batch_size,
                                            momentum_type=self.momentum_type,
                                            momentum=self.momentum,
                                            callback=self._print_dual_info,
                                            shuffle=self.shuffle,
                                            random_state=self.random_state,
                                            verbose=self.verbose).minimize()

        return self

    def callback(self, args=()):
        if self.f.primal.ndim == 2:
            self.x0_history.append(self.f.primal_x[0])
            self.x1_history.append(self.f.primal_x[1])
            self.f_x_history.append(-self.f.primal_f_x)
        if callable(self._callback):
            self._callback(self, *args, *self.callback_args)
