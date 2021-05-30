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
            raise ValueError('the upper bound must be > 0')
        self.ub = ub


class LagrangianQuadratic(Quadratic, ABC):
    """
    Abstract class for the lagrangian relaxation of a constrained quadratic function defined as:

            1/2 x^T Q x + q^T x
    """

    def __init__(self, primal, minres_verbose=False):
        if not isinstance(primal, Quadratic):
            raise TypeError(f'{primal} is not an allowed quadratic function')
        super().__init__(primal.Q, primal.q)
        self.primal = primal
        self.minres_verbose = minres_verbose
        try:
            self.L, self.low = cho_factor(self.Q)
            self.is_posdef = True
        except np.linalg.LinAlgError:
            self.is_posdef = False
        # multipliers constrained to be >= 0
        self.constrained_idx = np.full(self.ndim, False)
        # backup {lambda : x}
        self.last_lmbda = None
        self.last_x = None

    def f_star(self):
        return self.primal.function(self.x_star())

    def _solve_sym_nonposdef(self, Q, q):
        # since Q is not strictly psd, i.e., the function is linear along the eigenvectors
        # correspondent to the null eigenvalues, the system has infinite solutions so the
        # Lagrangian is non differentiable, and, for each solution x, the Lagrangian will
        # have a different subgradient; so we will choose the one that minimizes the
        # 2-norm since it is good almost like the gradient

        minres_verbose = self.minres_verbose() if callable(self.minres_verbose) else self.minres_verbose

        if minres_verbose:
            print('\n')

        # `min ||Qx - q||` is formally equivalent to solve the linear system:
        #                       (Q^T Q) x = (Q^T q)^T x

        Q, q = np.inner(Q, Q), Q.T.dot(q)

        x = minres(Q, q, show=minres_verbose)[0]

        return x

    def hessian(self, lmbda):
        H = np.zeros((self.ndim, self.ndim))
        H[0:self.Q.shape[0], 0:self.Q.shape[1]] = self.Q
        return H


class LagrangianEqualityConstrainedQuadratic(LagrangianQuadratic):
    """
    Construct the lagrangian dual relaxation of an equality constrained quadratic function defined as:

            1/2 x^T Q x + q^T x : A x = 0
    """

    def __init__(self, primal, A, minres_verbose=False):
        super().__init__(primal=primal,
                         minres_verbose=minres_verbose)
        self.A = np.asarray(A, dtype=float)
        A = np.atleast_2d(A)
        # self.Z = scipy.null_space(A)  # more complex, uses SVD
        Q, R = np.linalg.qr(A.T, mode='complete')
        # null space aka kernel - range aka image
        self.Z = Q[:, A.shape[0]:]  # orthonormal basis for the null space of A, i.e., ker(A) = im(Q)
        assert np.allclose(self.A.dot(self.Z), 0)
        self.proj_Q = self.Z.T.dot(self.Q).dot(self.Z)  # project
        try:
            self.L, self.low = cho_factor(self.proj_Q)
            self.is_posdef = True
        except np.linalg.LinAlgError:
            self.is_posdef = False

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  A=self.A,
                                  b=np.zeros(1),
                                  solver='cvxopt')
        return self.x_opt

    def function(self, mu):
        """
        Compute the value of the lagrangian relaxation defined as:

        L(x, mu) = 1/2 x^T Q x + q^T x - mu^T A x
         L(x, mu) = 1/2 x^T Q x + (q - mu A)^T x

        Taking the derivative of the Lagrangian wrt x and settings it to 0 gives:

                Q x + (q - mu A) = 0

        so, the optimal solution of the Lagrangian relaxation is the solution of the linear system:

                Q x = - (q - mu A)

        :param mu: the dual variable wrt evaluate the function
        :return: the function value wrt mu
        """
        ql = self.q - mu.dot(self.A)
        if np.array_equal(self.last_lmbda, mu):
            x = self.last_x.copy()  # speedup: just restore optimal solution
        else:
            proj_ql = self.Z.T.dot(ql)  # project
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -proj_ql)
            else:
                x = self._solve_sym_nonposdef(self.proj_Q, -proj_ql)
            x = self.Z.dot(x)  # recover the primal solution
            # backup new {lambda : x}
            self.last_lmbda = mu.copy()
            self.last_x = x.copy()
        return 0.5 * x.dot(self.Q).dot(x) + ql.dot(x)

    def jacobian(self, mu):
        """
        With x optimal solution of the minimization problem, the jacobian
        of the Lagrangian dual relaxation at mu is:

                                [-A x]

        However, we rather want to maximize the Lagrangian dual relaxation,
        hence we have to change the sign of gradient entries:

                                 [A x]

        :param mu: the dual variable wrt evaluate the gradient
        :return: the gradient wrt mu
        """
        if np.array_equal(self.last_lmbda, mu):
            x = self.last_x.copy()  # speedup: just restore optimal solution
        else:
            ql = self.q - mu.dot(self.A)
            proj_ql = self.Z.T.dot(ql)  # project
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -proj_ql)
            else:
                x = self._solve_sym_nonposdef(self.proj_Q, -proj_ql)
            x = self.Z.dot(x)  # recover the primal solution
            # backup new {lambda : x}
            self.last_lmbda = mu.copy()
            self.last_x = x.copy()
        return self.A * x


class LagrangianLowerBoundedQuadratic(LagrangianQuadratic):
    """
    Construct the lagrangian relaxation of a lower bounded quadratic function defined as:

            1/2 x^T Q x + q^T x : x >= 0
    """

    def __init__(self, primal, minres_verbose=False):
        super().__init__(primal=primal,
                         minres_verbose=minres_verbose)
        self.lb = np.zeros(self.ndim)
        self.constrained_idx = np.full(self.ndim, True)  # lambda >= 0

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  lb=np.zeros_like(self.q),
                                  solver='cvxopt')
        return self.x_opt

    def function(self, lmbda):
        """
        The Lagrangian relaxation is defined as:

         L(x, lambda) = 1/2 x^T Q x + q^T x - lambda^T x
          L(x, lambda) = 1/2 x^T Q x + (q - lambda)^T x

        where lambda is constrained to be >= 0.

        Taking the derivative of the Lagrangian wrt x and settings it to 0 gives:

                Q x + (q - lambda) = 0

        so, the optimal solution of the Lagrangian relaxation is the solution of the linear system:

                Q x = - (q - lambda)

        :param lmbda: the dual variable wrt evaluate the function
        :return: the function value wrt lambda
        """
        ql = self.q - lmbda
        if np.array_equal(self.last_lmbda, lmbda):
            x = self.last_x.copy()  # speedup: just restore optimal solution
        else:
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -ql)
            else:
                x = self._solve_sym_nonposdef(self.Q, -ql)
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return 0.5 * x.dot(self.Q).dot(x) + ql.dot(x)

    def jacobian(self, lmbda):
        """
        With x optimal solution of the minimization problem, the jacobian
        of the Lagrangian dual relaxation at lambda is:

                                [-x]

        However, we rather want to maximize the Lagrangian dual relaxation,
        hence we have to change the sign of gradient entries:

                                 [x]

        :param lmbda: the dual variable wrt evaluate the gradient
        :return: the gradient wrt lambda
        """
        if np.array_equal(self.last_lmbda, lmbda):
            x = self.last_x.copy()  # speedup: just restore optimal solution
        else:
            ql = self.q - lmbda
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -ql)
            else:
                x = self._solve_sym_nonposdef(self.Q, -ql)
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return x


class LagrangianEqualityLowerBoundedQuadratic(LagrangianEqualityConstrainedQuadratic):
    """
    Construct the lagrangian relaxation of an equality constrained
    quadratic function with lower bounds defined as:

            1/2 x^T Q x + q^T x : A x = 0, x >= 0
    """

    def __init__(self, primal, A, minres_verbose=False):
        super().__init__(primal=primal,
                         A=A,
                         minres_verbose=minres_verbose)
        self.lb = np.zeros(self.ndim)
        self.ndim *= 2
        self.constrained_idx = np.hstack((np.full(int(self.ndim / 2), False),  # mu
                                          np.full(int(self.ndim / 2), True)))  # lambda >= 0

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

        where mu are the first n components of lambda which controls the equality
        constraints and lambda are the last n components which controls the inequality
        constraint which are constrained to be >= 0.

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
            proj_ql = self.Z.T.dot(ql)  # project
            # if self.is_posdef:
            #     x = cho_solve((self.L, self.low), -proj_ql)
            # else:
            x = self._solve_sym_nonposdef(self.proj_Q, -proj_ql)
            x = self.Z.dot(x)  # recover the primal solution
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
            proj_ql = self.Z.T.dot(ql)  # project
            # if self.is_posdef:
            #     x = cho_solve((self.L, self.low), -proj_ql)
            # else:
            x = self._solve_sym_nonposdef(self.proj_Q, -proj_ql)
            x = self.Z.dot(x)  # recover the primal solution
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return np.hstack((self.A * x, x))


class LagrangianBoxConstrainedQuadratic(LagrangianLowerBoundedQuadratic):
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
        self.lb = np.zeros_like(ub)
        self.constrained_idx = np.full(self.ndim, True)  # lambda_+, lambda_- >= 0

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  lb=np.zeros_like(self.ub),
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
                x = self._solve_sym_nonposdef(self.Q, -ql)
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
                x = self._solve_sym_nonposdef(self.Q, -ql)
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return np.hstack((self.ub - x, x))


class LagrangianEqualityBoxConstrainedQuadratic(LagrangianEqualityConstrainedQuadratic):
    """
    Construct the lagrangian dual relaxation of an equality constrained
    quadratic function with lower and upper bounds defined as:

            1/2 x^T Q x + q^T x : A x = 0, lb = 0 <= x <= ub
    """

    def __init__(self, primal, A, ub, minres_verbose=False):
        super().__init__(primal=primal,
                         A=A,
                         minres_verbose=minres_verbose)
        self.ndim *= 3
        if any(u < 0 for u in ub):
            raise ValueError('the lower bound must be > 0')
        self.ub = np.asarray(ub, dtype=float)
        self.lb = np.zeros_like(ub)
        self.constrained_idx = np.hstack((np.full(int(self.ndim / 3), False),  # mu
                                          np.full(int(self.ndim / 3) * 2, True)))  # lambda_+, lambda_ >= 0

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  A=self.A,
                                  b=np.zeros(1),
                                  lb=np.zeros_like(self.ub),
                                  ub=self.ub,
                                  solver='cvxopt')
        return self.x_opt

    def function(self, lmbda):
        """
        Compute the value of the lagrangian relaxation defined as:

        L(x, mu, lambda_+, lambda_-) = 1/2 x^T Q x + q^T x - mu^T A x - lambda_+^T (ub - x) - lambda_-^T x
        L(x, mu, lambda_+, lambda_-) = 1/2 x^T Q x + (q - mu A + lambda_+ - lambda_-)^T x - lambda_+^T ub

        where mu are the first n components of lambda which controls the equality constraints and
        lambda_+^T are the second n components of lambda and lambda_-^T are the last n components
        which controls the inequality constraint which are constrained to be >= 0.

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
            proj_ql = self.Z.T.dot(ql)  # project
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -proj_ql)
            else:
                x = self._solve_sym_nonposdef(self.proj_Q, -proj_ql)
            x = self.Z.dot(x)  # recover the primal solution
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
            proj_ql = self.Z.T.dot(ql)  # project
            if self.is_posdef:
                x = cho_solve((self.L, self.low), -proj_ql)
            else:
                x = self._solve_sym_nonposdef(self.proj_Q, -proj_ql)
            x = self.Z.dot(x)  # recover the primal solution
            # backup new {lambda : x}
            self.last_lmbda = lmbda.copy()
            self.last_x = x.copy()
        return np.hstack((self.A * x, self.ub - x, x))
