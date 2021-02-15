import numpy as np

from . import LagrangianBoxConstrainedQuadratic
from .. import Optimizer
from ..unconstrained import ProximalBundle
from ..unconstrained.line_search import LineSearchOptimizer
from ..unconstrained.line_search.line_search import LagrangianArmijoWolfeLineSearch
from ..unconstrained.stochastic import StochasticOptimizer, AdaGrad, StochasticMomentumOptimizer


class LagrangianDual(Optimizer):

    def __init__(self,
                 f,
                 x=np.zeros,
                 optimizer=AdaGrad,
                 eps=1e-6,
                 step_size=0.01,
                 momentum_type='none',
                 momentum=0.9,
                 max_iter=1000,
                 max_f_eval=1000,
                 callback=None,
                 callback_args=(),
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 dual_solver='cg',
                 dual_verbose=False,
                 verbose=False):
        super().__init__(f=f,
                         x=x,
                         eps=eps,
                         max_iter=max_iter,
                         callback=callback,
                         callback_args=callback_args,
                         verbose=verbose)
        if not isinstance(f, LagrangianBoxConstrainedQuadratic):
            raise TypeError(f'{f} is not an allowed constrained quadratic optimization function')
        self.optimizer = optimizer
        self.step_size = step_size
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.max_f_eval = max_f_eval
        self.mu = mu
        self.master_solver = master_solver
        self.master_verbose = master_verbose
        self.dual_solver = dual_solver
        self.dual_verbose = dual_verbose
        # initialize the primal problem
        self.primal_x = self.f.ub / 2  # starts from the middle of the box
        self.primal_f_x = self.f.primal.function(self.primal_x)
        if self.f.primal.ndim == 2:
            self.x0_history = [self.primal_x[0]]
            self.x1_history = [self.primal_x[1]]
            self.f_x_history = [self.primal_f_x]

    def minimize(self):

        self.f.verbose = lambda: self.is_verbose() and self.dual_verbose
        self.f.solver = self.dual_solver

        if issubclass(self.optimizer, LineSearchOptimizer):

            self.optimizer = self.optimizer(f=self.f,
                                            x=self.x,
                                            max_iter=self.max_iter,
                                            max_f_eval=self.max_f_eval,
                                            callback=self._update_primal_dual,
                                            verbose=self.verbose)
            self.optimizer.line_search = LagrangianArmijoWolfeLineSearch(self.optimizer.f,
                                                                         self.optimizer.line_search.max_f_eval,
                                                                         self.optimizer.line_search.m1,
                                                                         self.optimizer.line_search.m2,
                                                                         self.optimizer.line_search.a_start,
                                                                         self.optimizer.line_search.tau,
                                                                         self.optimizer.line_search.sfgrd,
                                                                         self.optimizer.line_search.min_a)


        elif issubclass(self.optimizer, StochasticOptimizer):

            if issubclass(self.optimizer, StochasticMomentumOptimizer):

                self.optimizer = self.optimizer(f=self.f,
                                                x=self.x,
                                                step_size=self.step_size,
                                                epochs=self.max_iter,
                                                momentum_type=self.momentum_type,
                                                momentum=self.momentum,
                                                callback=self._update_primal_dual,
                                                verbose=self.verbose)

            else:

                self.optimizer = self.optimizer(f=self.f,
                                                x=self.x,
                                                step_size=self.step_size,
                                                epochs=self.max_iter,
                                                callback=self._update_primal_dual,
                                                verbose=self.verbose)

        elif issubclass(self.optimizer, ProximalBundle):

            self.optimizer = self.optimizer(f=self.f,
                                            x=self.x,
                                            mu=self.mu,
                                            max_iter=self.max_iter,
                                            callback=self._update_primal_dual,
                                            lagrangian=True,
                                            master_solver=self.master_solver,
                                            master_verbose=self.master_verbose,
                                            verbose=self.verbose)

        self.__dict__.update(self.optimizer.minimize().__dict__)
        # assert np.all(self.x >= 0)
        return self

    def _update_primal_dual(self, opt):

        if not isinstance(opt, ProximalBundle):
            # project the direction over the active constraints
            opt.g_x[np.logical_and(opt.x <= 1e-12, -opt.g_x < 0)] = 0

        # compute an heuristic solution out of the solution x of
        # the Lagrangian relaxation by projecting x on the box
        self.f.last_x[self.f.last_x < 0] = 0
        idx = self.f.last_x > self.f.ub
        self.f.last_x[idx] = self.f.ub[idx]

        v = self.f.primal.function(self.f.last_x)
        if v < self.primal_f_x:
            self.primal_x = self.f.last_x
            self.primal_f_x = v

        gap = (self.primal_f_x - opt.f_x) / max(abs(self.primal_f_x), 1)

        if opt.is_verbose():
            print('\tpcost: {: 1.4e}'.format(self.primal_f_x), end='')
            print('\tgap: {: 1.4e}'.format(gap), end='')
            if not self.f.is_posdef:
                if self.f.last_itn:
                    print('\titn: {:3d}'.format(self.f.last_itn), end='')
                print('\trnorm: {:1.4e}'.format(self.f.last_rnorm), end='')

        self.callback(self.callback_args)

        if gap <= self.eps:
            opt.status = 'optimal'
            raise StopIteration

        self.iter += 1

    def callback(self, args=()):
        if self.f.primal.ndim == 2:
            self.x0_history.append(self.primal_x[0])
            self.x1_history.append(self.primal_x[1])
            self.f_x_history.append(self.primal_f_x)
        if callable(self._callback):
            self._callback(*self.callback_args)
