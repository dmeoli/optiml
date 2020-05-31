import numpy as np
from scipy.linalg import ldl

from .. import Optimizer
from .. import Quadratic
from ..unconstrained import ProximalBundle
from ..unconstrained.line_search import LineSearchOptimizer
from ..unconstrained.stochastic import StochasticOptimizer, AdaGrad
from ..utils import ldl_solve


class LagrangianDual(Optimizer):

    def __init__(self, dual, optimizer=AdaGrad, eps=1e-6, step_size=0.01, momentum_type='none', momentum=0.9,
                 batch_size=None, max_iter=1000, max_f_eval=1000, callback=None, callback_args=(),
                 master_solver='ecos', master_verbose=False, shuffle=True, random_state=None, verbose=False):
        super().__init__(f=dual, x=np.zeros(dual.ndim), eps=eps, max_iter=max_iter,
                         callback=callback, callback_args=callback_args, verbose=verbose)
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
        self.master_solver = master_solver
        self.master_verbose = master_verbose
        self.shuffle = shuffle
        self.random_state = random_state

    def _print_dual_info(self, opt):
        gap = (self.f.primal_value - self.f_x) / max(abs(self.f_x), 1)

        if ((isinstance(opt, LineSearchOptimizer) and opt.is_verbose()) or
                (isinstance(opt, StochasticOptimizer) and opt.is_batch_end())):
            print('\tub: {: 1.4e}'.format(self.f.primal_value), end='')
            print(' - pcost: {: 1.4e}'.format(self.f_x), end='')
            print(' - gap: {: 1.4e}'.format(gap), end='')

        self.callback()

        if gap <= self.eps:
            self.status = 'optimal'
            raise StopIteration

        self.x, self.f_x = self.f.primal_solution, self.f.primal_value

        self.iter += 1

    def minimize(self):

        self.f_x = self.f.function(self.x)

        if issubclass(self.optimizer, LineSearchOptimizer):

            self.optimizer = self.optimizer(f=self.f, x=self.x, max_iter=self.max_iter, max_f_eval=self.max_f_eval,
                                            callback=self._print_dual_info, verbose=self.verbose).minimize()

        elif issubclass(self.optimizer, StochasticOptimizer):

            self.optimizer = self.optimizer(f=self.f, x=self.x, step_size=self.step_size, epochs=self.max_iter,
                                            batch_size=self.batch_size, momentum_type=self.momentum_type,
                                            momentum=self.momentum, callback=self._print_dual_info,
                                            shuffle=self.shuffle, random_state=self.random_state,
                                            verbose=self.verbose).minimize()

        elif issubclass(self.optimizer, ProximalBundle):

            self.optimizer = self.optimizer(f=self.f, x=self.x, max_iter=self.max_iter,
                                            master_solver=self.master_solver, verbose=self.verbose,
                                            master_verbose=self.master_verbose).minimize()

        return self

    def callback(self, args=()):
        if self.f.primal.ndim == 2:
            self.x0_history.append(self.f.primal_solution[0])
            self.x1_history.append(self.f.primal_solution[1])
            self.f_x_history.append(self.f.primal_value)
        if callable(self._callback):
            self._callback(self, *args, *self.callback_args)


class LagrangianConstrainedQuadratic(Quadratic):

    def __init__(self, quad, A, ub):
        """
        Construct the lagrangian relaxation of a constrained quadratic function defined as:

                1/2 x^T Q x + q^T x : A x = 0, 0 <= x <= ub

        :param quad: constrained quadratic function to be relaxed
        """
        if not isinstance(quad, Quadratic):
            raise TypeError(f'{quad} is not an allowed quadratic function')
        super().__init__(quad.Q, quad.q)
        self.ndim *= 3
        # Compute the LDL^T Cholesky symmetric indefinite factorization
        # of Q because it is symmetric but could be not positive definite.
        # This will be used at each iteration to solve the Lagrangian relaxation.
        self.L, self.D, self.P = ldl(self.Q)
        self.A = np.asarray(A, dtype=np.float)
        if any(u < 0 for u in ub):
            raise ValueError('the lower bound must be > 0')
        self.ub = np.asarray(ub, dtype=np.float)
        self.primal = quad
        self.primal_solution = np.inf
        self.primal_value = np.inf

    def x_star(self):
        raise np.full(fill_value=np.nan, shape=self.ndim)

    def f_star(self):
        return np.inf

    def function(self, lmbda):
        """
        Compute the value of the lagrangian relaxation defined as:

             L(x, lambda) = 1/2 x^T Q x + q^T x - mu^T A x - lambda_+^T (ub - x) - lambda_^T- x
           L(x, lambda) = 1/2 x^T Q x + (q^T - mu^T A + lambda_+^T - lambda_-^T) x - lambda_+^T ub

        where mu are the first n components of lambda which controls the equality constraints,
        lambda_+^T are the second n components of lambda and lambda_-^T are the last n components;
        both controls the box-constraints and are constrained to be >= 0.

        The optimal solution of the Lagrangian relaxation is the unique solution of the linear system:

                Q x = q^T - mu^T A + lambda_+^T - lambda_-^T

        Since we have saved the LDL^T Cholesky factorization of Q,
        i.e., Q = L D L^T, we obtain this by solving:

             L D L^T x = q^T - mu^T A + lambda_+^T - lambda_-^T

        :param lmbda:
        :return: the function value
        """
        mu, lmbda_p, lmbda_n = np.split(lmbda, 3)
        ql = self.q.T - mu.T.dot(self.A) + lmbda_p.T - lmbda_n.T
        x = ldl_solve((self.L, self.D, self.P), -ql)
        return (0.5 * x.T.dot(self.Q) + ql.T).dot(x) - lmbda_p.T.dot(self.ub)

    def jacobian(self, lmbda):
        """
        Compute the jacobian of the lagrangian relaxation as follow: with x the optimal
        solution of the minimization problem, the gradient at lambda is:

                                [-A x, x - ub, -x]

        However, we rather want to maximize the lagrangian relaxation, hence we have to
        change the sign of both function values and gradient entries:

                                 [A x, ub - x, x]
        :param lmbda:
        :return:
        """
        mu, lmbda_p, lmbda_n = np.split(lmbda, 3)
        ql = self.q.T - mu.T.dot(self.A) + lmbda_p.T - lmbda_n.T
        x = ldl_solve((self.L, self.D, self.P), -ql)
        g = np.hstack((self.A * x, self.ub - x, x))

        # compute an heuristic solution out of the solution x of
        # the Lagrangian relaxation by projecting x on the box
        x[x < 0] = 0
        idx = x > self.ub
        x[idx] = self.ub[idx]

        v = self.primal.function(x)
        if v < self.primal_value:
            self.primal_solution = x
            self.primal_value = -v

        return g
