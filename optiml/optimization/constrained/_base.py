import numpy as np
from scipy.linalg import ldl

from ..unconstrained import Quadratic
from ..utils import ldl_solve


class LagrangianBoxConstrainedQuadratic(Quadratic):

    def __init__(self, quad, ub):
        """
        Construct the lagrangian relaxation of a constrained quadratic function defined as:
                           
                           1/2 x^T Q x + q^T x : 0 <= x <= ub
                           
                       1/2 x^T Q x + q^T x - lambda^+ (ub - x) - lambda^- x
                    1/2 x^T Q x + (q^T + lambda^+ - lambda^-) x - lambda^+ ub

        where lambda^+ are the first n components of lambda, and lambda^- the last n components;
        both are constrained to be >= 0.
        :param quad: constrained quadratic function to be relaxed
        :param ub: upper bounds vector
        """
        if not isinstance(quad, Quadratic):
            raise TypeError(f'{quad} is not an allowed quadratic function')
        super().__init__(quad.Q, quad.q)
        self.ndim *= 2
        self.ub = ub
        # Compute the LDL^T Cholesky symmetric indefinite factorization
        # of Q because it is symmetric but could be not positive definite.
        # This will be used at each iteration to solve the Lagrangian relaxation.
        self.L, self.D, self.P = ldl(self.Q)
        self.primal = quad
        self.primal_solution = np.inf
        self.primal_value = np.inf

    def x_star(self):
        raise np.full(fill_value=np.nan, shape=self.ndim)

    def f_star(self):
        return np.inf

    def function(self, lmbda):
        """
        The optimal solution of the Lagrangian relaxation is the unique
        solution of the linear system:

                        Q x = q^T + lambda^+ - lambda^-

        Since we have saved the LDL^T Cholesky factorization of Q,
        i.e., Q = L D L^T, we obtain this by solving:

                     L D L^T x = q^T + lambda^+ - lambda^-

        :param lmbda:
        :return: the function value
        """
        ql = self.q.T + lmbda[:self.primal.ndim] - lmbda[self.primal.ndim:]
        x = ldl_solve((self.L, self.D, self.P), -ql)
        return (0.5 * x.T.dot(self.Q) + ql.T).dot(x) - lmbda[:self.primal.ndim].T.dot(self.ub)

    def jacobian(self, lmbda):
        """
        Compute the jacobian of the lagrangian relaxation as follow: with x the optimal
        solution of the minimization problem, the gradient at lambda is [x - u, -x].
        However, we rather want to maximize the lagrangian relaxation, hence we have to
        change the sign of both function values and gradient entries.
        :param lmbda:
        :return:
        """
        ql = self.q.T + lmbda[:self.primal.ndim] - lmbda[self.primal.ndim:]
        x = ldl_solve((self.L, self.D, self.P), -ql)
        g = np.hstack((self.ub - x, x))

        # compute an heuristic solution out of the solution x of
        # the Lagrangian relaxation by projecting x on the box
        x[x < 0] = 0
        idx = x > self.ub
        x[idx] = self.ub[idx]

        v = self.primal.function(x)
        if v < self.primal_value:
            self.primal_solution = x
            self.primal_value = v

        return g
