from abc import ABC

import numpy as np
from qpsolvers import solve_qp
from scipy.linalg import cho_solve, cho_factor
from scipy.sparse.linalg import minres

from optiml.opti import Optimizer, Quadratic


class BoxConstrainedQuadraticOptimizer(Optimizer, ABC):

    def __init__(self,
                 quad,
                 ub,
                 x=None,
                 eps=1e-6,
                 max_iter=1000,
                 callback=None,
                 callback_args=(),
                 verbose=False):
        if not isinstance(quad, Quadratic):
            raise TypeError(f'{quad} is not an allowed quadratic function')
        super().__init__(f=quad,
                         x=x or ub / 2,  # starts from the middle of the box
                         eps=eps,
                         max_iter=max_iter,
                         callback=callback,
                         callback_args=callback_args,
                         verbose=verbose)
        if any(u < 0 for u in ub):
            raise ValueError('the lower bound must be > 0')
        self.ub = ub


class LagrangianQuadratic(Quadratic):
    """
    Construct the lagrangian relaxation of a constrained quadratic function defined as:

            1/2 x^T Q x + q^T x : x >= 0
    """

    def __init__(self, primal, minres_verbose=False):
        if not isinstance(primal, Quadratic):
            raise TypeError(f'{primal} is not an allowed quadratic function')
        super().__init__(primal.Q, primal.q)
        self.primal = primal
        try:
            self.L, self.low = cho_factor(self.Q)
            self.is_posdef = True
        except np.linalg.LinAlgError:
            self.is_posdef = False
        self.minres_verbose = minres_verbose
        # backup {lambda : x}
        self.last_lmbda = None
        self.last_x = None
        self.last_r1norm = None

    def _solve_sym_nonposdef(self, ql):
        # since Q is indefinite, i.e., the function is linear along the eigenvectors
        # correspondent to the null eigenvalues, the system has not solutions, so we
        # will choose the one that minimizes the residue in the least squares sense
        # see more @ https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems

        minres_verbose = self.minres_verbose() if callable(self.minres_verbose) else self.minres_verbose

        if minres_verbose:
            print('\n')

        # `min ||Qx - ql||^2` is formally equivalent to solve the linear system:
        #                           Q^T Q x = Q^T ql

        Q, ql = np.inner(self.Q, self.Q), self.Q.T.dot(ql)

        x = minres(Q, -ql, show=minres_verbose)[0]

        self.last_rnorm = np.linalg.norm(-ql - Q.dot(x))  # || ql - Qx ||

        return x

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  lb=np.zeros_like(self.q),
                                  solver='cvxopt')
        return self.x_opt

    def f_star(self):
        return self.primal.function(self.x_star())

    def function(self, x):
        self.last_x = x.copy()
        return super().function(x)

    def jacobian(self, x):
        self.last_x = x.copy()
        return super().jacobian(x)


class LagrangianEqualityConstrainedQuadratic(LagrangianQuadratic):
    """
    Construct the lagrangian relaxation of a constrained quadratic function defined as:

            1/2 x^T Q x + q^T x : A x = 0, x >= 0
    """

    def __init__(self, primal, A, minres_verbose=False):
        super().__init__(primal=primal,
                         minres_verbose=minres_verbose)
        self.ndim *= 2
        self.A = np.asarray(A, dtype=float)

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  A=self.A,
                                  b=np.zeros(1),
                                  lb=np.zeros_like(self.q),
                                  solver='cvxopt')
        return self.x_opt

    def function(self, lmbda):
        """
        Compute the value of the lagrangian relaxation defined as:

        L(x, mu, lambda) = 1/2 x^T Q x + q^T x - mu^T A x - lambda^T x
        L(x, mu, lambda) = 1/2 x^T Q x + (q - mu A - lambda)^T x

        where mu are the first n components of lambda which controls the equality constraints and
        lambda are the last n components which controls the inequality constraint;
        both are constrained to be >= 0.

        Taking the derivative of the Lagrangian wrt x and settings it to 0 gives:

                Q x + (q - mu A - lambda) = 0

        so, the optimal solution of the Lagrangian relaxation is the solution of the linear system:

                Q x = - (q - mu A - lambda)

        :param lmbda: the dual variable wrt evaluate the function
        :return: the function value wrt lambda
        """
        mu, lmbda = np.split(lmbda, 2)
        ql = self.q - mu.dot(self.A) - lmbda
        if np.array_equal(self.last_lmbda, lmbda):
            x = self.last_x.copy()  # speedup: just restore optimal solution
        else:
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -ql)
            else:
                x = self._solve_sym_nonposdef(ql)
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return 0.5 * x.dot(self.Q).dot(x) + ql.dot(x)

    def jacobian(self, lmbda):
        """
        With x optimal solution of the minimization problem, the jacobian
        of the Lagrangian dual relaxation at lambda is:

                                [-A x, -x]

        However, we rather want to maximize the Lagrangian dual relaxation,
        hence we have to change the sign of gradient entries:

                                 [A x, x]

        :param lmbda: the dual variable wrt evaluate the gradient
        :return: the gradient wrt lambda
        """
        if np.array_equal(self.last_lmbda, lmbda):
            x = self.last_x.copy()  # speedup: just restore optimal solution
        else:
            mu, lmbda = np.split(lmbda, 2)
            ql = self.q - mu.dot(self.A) - lmbda
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -ql)
            else:
                x = self._solve_sym_nonposdef(ql)
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return np.hstack((self.A * x, x))

    def hessian(self, lmbda):
        H = np.zeros((self.ndim, self.ndim))
        H[0:self.Q.shape[0], 0:self.Q.shape[1]] = self.Q
        return H


class LagrangianBoxConstrainedQuadratic(LagrangianQuadratic):
    """
    Construct the Lagrangian dual relaxation of a box-constrained quadratic function defined as:

                    1/2 x^T Q x + q^T x : lb = 0 <= x <= ub
    """

    def __init__(self, primal, ub, minres_verbose=False):
        super().__init__(primal=primal,
                         minres_verbose=minres_verbose)
        self.ndim *= 2
        if any(u < 0 for u in ub):
            raise ValueError('the lower bound must be > 0')
        self.ub = np.asarray(ub, dtype=float)

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  lb=np.zeros_like(self.q),
                                  ub=self.ub,
                                  solver='cvxopt')
        return self.x_opt

    def function(self, lmbda):
        """
        The Lagrangian relaxation is defined as:

         L(x, lambda_+, lambda_-) = 1/2 x^T Q x + q^T x - lambda_+^T (ub - x) - lambda_-^T x
        L(x, lambda_+, lambda_-) = 1/2 x^T Q x + (q + lambda_+ - lambda_-)^T x - lambda_+^T ub

        where lambda_+ are the first n components of lambda and lambda_- are the last n
        components; both controls the box-constraints and are constrained to be >= 0.

        Taking the derivative of the Lagrangian wrt x and settings it to 0 gives:

                Q x + (q + lambda_+ - lambda_-) = 0

        so, the optimal solution of the Lagrangian relaxation is the solution of the linear system:

                Q x = - (q + lambda_+ - lambda_-)

        :param lmbda: the dual variable wrt evaluate the function
        :return: the function value wrt lambda
        """
        lmbda_p, lmbda_n = np.split(lmbda, 2)
        ql = self.q + lmbda_p - lmbda_n
        if np.array_equal(self.last_lmbda, lmbda):
            x = self.last_x.copy()  # speedup: just restore optimal solution
        else:
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -ql)
            else:
                x = self._solve_sym_nonposdef(ql)
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return 0.5 * x.dot(self.Q).dot(x) + ql.dot(x) - lmbda_p.dot(self.ub)

    def jacobian(self, lmbda):
        """
        With x optimal solution of the minimization problem, the jacobian
        of the Lagrangian dual relaxation at lambda is:

                                [x - ub, -x]

        However, we rather want to maximize the Lagrangian dual relaxation,
        hence we have to change the sign of gradient entries:

                                 [ub - x, x]

        :param lmbda: the dual variable wrt evaluate the gradient
        :return: the gradient wrt lambda
        """
        if np.array_equal(self.last_lmbda, lmbda):
            x = self.last_x.copy()  # speedup: just restore optimal solution
        else:
            lmbda_p, lmbda_n = np.split(lmbda, 2)
            ql = self.q + lmbda_p - lmbda_n
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -ql)
            else:
                x = self._solve_sym_nonposdef(ql)
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return np.hstack((self.ub - x, x))

    def hessian(self, lmbda):
        H = np.zeros((self.ndim, self.ndim))
        H[0:self.Q.shape[0], 0:self.Q.shape[1]] = self.Q
        return H


class LagrangianEqualityBoxConstrainedQuadratic(LagrangianBoxConstrainedQuadratic):
    """
    Construct the lagrangian relaxation of a constrained quadratic function defined as:

            1/2 x^T Q x + q^T x : A x = 0, lb = 0 <= x <= ub
    """

    def __init__(self, primal, A, ub, minres_verbose=False):
        super().__init__(primal=primal,
                         ub=ub,
                         minres_verbose=minres_verbose)
        self.ndim += int(self.ndim / 2)
        self.A = np.asarray(A, dtype=float)

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  A=self.A,
                                  b=np.zeros(1),
                                  lb=np.zeros_like(self.q),
                                  ub=self.ub,
                                  solver='cvxopt')
        return self.x_opt

    def function(self, lmbda):
        """
        Compute the value of the lagrangian relaxation defined as:

        L(x, mu, lambda_+, lambda_-) = 1/2 x^T Q x + q^T x - mu^T A x - lambda_+^T (ub - x) - lambda_-^T x
        L(x, mu, lambda_+, lambda_-) = 1/2 x^T Q x + (q - mu A + lambda_+ - lambda_-)^T x - lambda_+^T ub

        where mu are the first n components of lambda which controls the equality constraints,
        lambda_+^T are the second n components of lambda and lambda_-^T are the last n components;
        both controls the box-constraints and are constrained to be >= 0.

        Taking the derivative of the Lagrangian wrt x and settings it to 0 gives:

                Q x + (q - mu A + lambda_+ - lambda_-) = 0

        so, the optimal solution of the Lagrangian relaxation is the solution of the linear system:

                Q x = - (q - mu A + lambda_+ - lambda_-)

        :param lmbda: the dual variable wrt evaluate the function
        :return: the function value wrt lambda
        """
        mu, lmbda_p, lmbda_n = np.split(lmbda, 3)
        ql = self.q - mu.dot(self.A) + lmbda_p - lmbda_n
        if np.array_equal(self.last_lmbda, lmbda):
            x = self.last_x.copy()  # speedup: just restore optimal solution
        else:
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -ql)
            else:
                x = self._solve_sym_nonposdef(ql)
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return 0.5 * x.dot(self.Q).dot(x) + ql.dot(x) - lmbda_p.dot(self.ub)

    def jacobian(self, lmbda):
        """
        With x optimal solution of the minimization problem, the jacobian
        of the Lagrangian dual relaxation at lambda is:

                                [-A x, x - ub, -x]

        However, we rather want to maximize the Lagrangian dual relaxation,
        hence we have to change the sign of gradient entries:

                                 [A x, ub - x, x]

        :param lmbda: the dual variable wrt evaluate the gradient
        :return: the gradient wrt lambda
        """
        if np.array_equal(self.last_lmbda, lmbda):
            x = self.last_x.copy()  # speedup: just restore optimal solution
        else:
            mu, lmbda_p, lmbda_n = np.split(lmbda, 3)
            ql = self.q - mu.dot(self.A) + lmbda_p - lmbda_n
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -ql)
            else:
                x = self._solve_sym_nonposdef(ql)
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return np.hstack((self.A * x, self.ub - x, x))
