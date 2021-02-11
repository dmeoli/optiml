from abc import ABC

import numpy as np
from scipy.sparse.linalg import lsqr

from optiml.opti import Optimizer
from optiml.opti import Quadratic
from optiml.opti.utils import cholesky_solve


class BoxConstrainedQuadraticOptimizer(Optimizer, ABC):

    def __init__(self,
                 f,
                 ub,
                 eps=1e-6,
                 max_iter=1000,
                 callback=None,
                 callback_args=(),
                 verbose=False):
        if not isinstance(f, Quadratic):
            raise TypeError(f'{f} is not an allowed quadratic function')
        super().__init__(f=f,
                         x=ub / 2,  # starts from the middle of the box
                         eps=eps,
                         max_iter=max_iter,
                         callback=callback,
                         callback_args=callback_args,
                         verbose=verbose)
        if any(u < 0 for u in ub):
            raise ValueError('the lower bound must be > 0')
        self.ub = ub


class LagrangianBoxConstrainedQuadratic(Quadratic):
    """
    Construct the Lagrangian dual relaxation of a box-constrained quadratic function defined as:

                    1/2 x^T Q x + q^T x : 0 <= x <= ub
    """

    def __init__(self, quad, ub):
        if not isinstance(quad, Quadratic):
            raise TypeError(f'{quad} is not an allowed quadratic function')
        super().__init__(quad.Q, quad.q)
        self.primal = quad
        self.ndim *= 2
        try:
            self.L = np.linalg.cholesky(self.Q)
            self.is_posdef = True
        except np.linalg.LinAlgError:
            self.is_posdef = False
        if any(u < 0 for u in ub):
            raise ValueError('the lower bound must be > 0')
        self.ub = np.asarray(ub, dtype=float)
        # backup {lambda : x}
        self.last_lmbda = None
        self.last_x = None
        self.last_itn = None
        self.last_r1norm = None

    def x_star(self):
        raise np.full(fill_value=np.nan, shape=self.ndim)

    def f_star(self):
        return np.inf

    def function(self, lmbda):
        """
        The Lagrangian relaxation is defined as:

         L(x, lambda) = 1/2 x^T Q x + q^T x - lambda_+^T (ub - x) - lambda_-^T x
        L(x, lambda) = 1/2 x^T Q x + (q + lambda_+ - lambda_-)^T x - lambda_+^T ub

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
        if self.is_posdef:
            x = cholesky_solve(self.L, -ql)
        else:
            # since Q is indefinite, i.e., the function is linear along the eigenvectors
            # correspondent to the null eigenvalues, the system has not solutions, so we
            # will choose the one that minimizes the residue in the least-squares sense
            # (waiting for 'symmlq' in scipy.sparse.linalg)
            x, self.last_itn, self.last_r1norm = map(lsqr(self.Q, -ql).__getitem__, [0, 2, 3])
        self.last_lmbda = lmbda
        self.last_x = x
        return 0.5 * x.T.dot(self.Q).dot(x) + ql.T.dot(x) - lmbda_p.T.dot(self.ub)

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
        lmbda_p, lmbda_n = np.split(lmbda, 2)
        ql = self.q + lmbda_p - lmbda_n
        if self.is_posdef:
            x = cholesky_solve(self.L, -ql)
        else:
            # since Q is indefinite, i.e., the function is linear along the eigenvectors
            # correspondent to the null eigenvalues, the system has not solutions, so we
            # will choose the one that minimizes the residue in the least-squares sense
            # (waiting for 'symmlq' in scipy.sparse.linalg)
            x, self.last_itn, self.last_r1norm = map(lsqr(self.Q, -ql).__getitem__, [0, 2, 3])
        self.last_lmbda = lmbda
        self.last_x = x
        return np.hstack((self.ub - x, x))

    def hessian(self, x):
        raise NotImplementedError


class LagrangianConstrainedQuadratic(LagrangianBoxConstrainedQuadratic):
    """
    Construct the lagrangian relaxation of a constrained quadratic function defined as:

            1/2 x^T Q x + q^T x : A x = 0, 0 <= x <= ub
    """

    def __init__(self, quad, A, ub):
        super().__init__(quad, ub)
        self.ndim += int(self.ndim / 2)
        self.A = np.asarray(A, dtype=float)

    def function(self, lmbda):
        """
        Compute the value of the lagrangian relaxation defined as:

        L(x, lambda, mu) = 1/2 x^T Q x + q^T x - mu^T A x - lambda_+^T (ub - x) - lambda_-^T x
        L(x, lambda, mu) = 1/2 x^T Q x + (q - mu A + lambda_+ - lambda_-)^T x - lambda_+^T ub

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
        if self.is_posdef:
            x = cholesky_solve(self.L, -ql)
        else:
            # since Q is indefinite, i.e., the function is linear along the eigenvectors
            # correspondent to the null eigenvalues, the system has not solutions, so we
            # will choose the one that minimizes the residue in the least-squares sense
            # (waiting for 'symmlq' in scipy.sparse.linalg)
            x, self.last_itn, self.last_r1norm = map(lsqr(self.Q, -ql).__getitem__, [0, 2, 3])
        self.last_lmbda = lmbda
        self.last_x = x
        return 0.5 * x.T.dot(self.Q).dot(x) + ql.T.dot(x) - lmbda_p.T.dot(self.ub)

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
        mu, lmbda_p, lmbda_n = np.split(lmbda, 3)
        ql = self.q - mu.dot(self.A) + lmbda_p - lmbda_n
        if self.is_posdef:
            x = cholesky_solve(self.L, -ql)
        else:
            # since Q is indefinite, i.e., the function is linear along the eigenvectors
            # correspondent to the null eigenvalues, the system has not solutions, so we
            # will choose the one that minimizes the residue in the least-squares sense
            # (waiting for 'symmlq' in scipy.sparse.linalg)
            x, self.last_itn, self.last_r1norm = map(lsqr(self.Q, -ql).__getitem__, [0, 2, 3])
        self.last_lmbda = lmbda
        self.last_x = x
        return np.hstack((self.A * x, self.ub - x, x))
