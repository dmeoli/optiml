import numpy as np

from .. import Optimizer, Quadratic
from ..unconstrained.line_search import LineSearchOptimizer
from ..unconstrained.stochastic import StochasticOptimizer, AdaGrad
from ..unconstrained.stochastic.schedules import constant
from ..utils import cholesky_solve, nearest_posdef


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
                 step_size_schedule=constant,
                 momentum_schedule=constant,
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
        self.step_size_schedule = step_size_schedule
        self.momentum_schedule = momentum_schedule
        self.batch_size = batch_size
        self.max_f_eval = max_f_eval
        self.shuffle = shuffle
        self.random_state = random_state

    def _print_dual_info(self, opt):
        gap = (self.f.primal_f_x - self.f_x) / max(abs(self.f_x), 1)

        if ((isinstance(opt, LineSearchOptimizer) and opt.is_verbose()) or
            (isinstance(opt, StochasticOptimizer) and opt.is_batch_end())) and self.is_verbose():
            print('\tub: {: 1.4e}'.format(self.f_x), end='')
            print(' - pcost: {: 1.4e}'.format(self.f.primal_f_x), end='')
            print(' - gap: {: 1.4e}'.format(gap), end='')

        self.callback()

        self.x, self.f_x = self.f.primal_x, self.f.primal_f_x

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
                                            step_size_schedule=self.step_size_schedule,
                                            epochs=self.max_iter,
                                            batch_size=self.batch_size,
                                            momentum_type=self.momentum_type,
                                            momentum=self.momentum,
                                            momentum_schedule=self.momentum_schedule,
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


class LagrangianConstrainedQuadratic(Quadratic):
    """
    Construct the lagrangian relaxation of a constrained quadratic function defined as:

            1/2 x^T Q x + q^T x : A x = 0, 0 <= x <= ub
    """

    def __init__(self, quad, A, ub):
        if not isinstance(quad, Quadratic):
            raise TypeError(f'{quad} is not an allowed quadratic function')
        super().__init__(nearest_posdef(quad.Q), quad.q)
        self.ndim *= 3
        self.L = np.linalg.cholesky(self.Q)
        self.A = np.asarray(A, dtype=np.float)
        if any(u < 0 for u in ub):
            raise ValueError('the lower bound must be > 0')
        self.ub = np.asarray(ub, dtype=np.float)
        self.primal = quad
        self.primal_x = np.inf
        self.primal_f_x = np.inf
        self.last_lmbda = None
        self.last_x = None

    def x_star(self):
        raise np.full(fill_value=np.nan, shape=self.ndim)

    def f_star(self):
        return np.inf

    def function(self, lmbda):
        """
        The Lagrangian relaxation is defined as:

        L(x, mu, lambda_+, lambda_-) = 1/2 x^T Q x + q^T x - mu^T A x - lambda_+^T (ub - x) - lambda_-^T x
        L(x, mu, lambda_+, lambda_-) = 1/2 x^T Q x + (q - mu A^T + lambda_+ - lambda_-)^T x - lambda_+^T ub

        where mu are the first n components of lambda, lambda_+^T are the second n components of lambda
        and lambda_-^T are the last n components; all are constrained to be >= 0.

        Taking the derivative of the Lagrangian primal L(x, mu, lambda_+, lambda_-) wrt x and settings it to 0 gives:

                Q x + q - mu A^T + lambda_+ - lambda_- = 0

        so, the optimal solution of the Lagrangian relaxation is the solution of the linear system:

                Q x = - q - mu A^T + lambda_+ - lambda_-

        :param lmbda: the dual variable wrt evaluate the function
        :return: the function value wrt lambda
        """
        mu, lmbda_p, lmbda_n = np.split(lmbda, 3)
        ql = self.q - mu.dot(self.A) + lmbda_p - lmbda_n
        if np.array_equal(lmbda, self.last_lmbda):
            x = self.last_x
        else:
            x = cholesky_solve(self.L, -ql)
            self.last_lmbda = lmbda
            self.last_x = x
        return (0.5 * x.T.dot(self.Q) + ql.T).dot(x) - lmbda_p.T.dot(self.ub)

    def jacobian(self, lmbda):
        """
        Compute the jacobian of the Lagrangian dual relaxation as follow: with x the optimal
        solution of the minimization problem, the gradient at lambda is:

                                [-A x, x - ub, -x]

        However, we rather want to maximize the Lagrangian dual relaxation, hence we have to
        change the sign of both function values and gradient entries:

                                 [A x, ub - x, x]

        :param lmbda: the dual variable wrt evaluate the gradient
        :return: the gradient wrt lambda
        """
        if np.array_equal(lmbda, self.last_lmbda):
            x = self.last_x
        else:
            mu, lmbda_p, lmbda_n = np.split(lmbda, 3)
            ql = self.q - mu.dot(self.A) + lmbda_p - lmbda_n
            x = cholesky_solve(self.L, -ql)
            self.last_lmbda = lmbda
            self.last_x = x
        g = np.hstack((self.A * x, self.ub - x, x))

        v = self.primal.function(x)
        if v < self.primal_f_x:
            self.primal_x = x
            self.primal_f_x = -v

        return g
