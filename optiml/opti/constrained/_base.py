from abc import ABC

import autograd.numpy as np
from qpsolvers import solve_qp, check_problem_constraints

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
        self.ub = np.asarray(ub, dtype=float)


class LagrangianQuadratic(Quadratic):
    """
    Abstract class for the lagrangian relaxation of a constrained quadratic function defined as:

            1/2 x^T Q x + q^T x : A x = b, G x <= h, lb <= x <= ub
    """

    def __init__(self, primal, A=None, b=None, G=None, h=None, lb=None, ub=None, rho=1.):
        if not isinstance(primal, Quadratic):
            raise TypeError(f'{primal} is not an allowed quadratic function')
        super().__init__(primal.Q, primal.q)
        self.primal = primal
        self.A = np.atleast_2d(A).astype(float) if A is not None else None
        self.b = b
        self.G = np.atleast_2d(G).astype(float) if G is not None else None
        self.h = h
        self.lb = np.asarray(lb, dtype=float) if lb is not None else None
        if self.lb is not None:
            if self.G is None:
                self.G = -np.eye(self.ndim)
                self.h = -self.lb
            else:
                self.G = np.concatenate((self.G, -np.eye(self.ndim)), axis=0)
                self.h = np.concatenate((self.h, -self.lb))
        self.ub = np.asarray(ub, dtype=float) if ub is not None else None
        if self.ub is not None:
            if self.G is None:
                self.G = np.eye(self.ndim)
                self.h = self.ub
            else:
                self.G = np.concatenate((self.G, np.eye(self.ndim)), axis=0)
                self.h = np.concatenate((self.h, self.ub))
        if not 0 < rho <= 1:
            raise ValueError('rho must be must be between 0 and 1')
        self.rho = rho
        if G is None and h is not None:
            raise ValueError("incomplete inequality constraint (missing G)")
        if G is not None and h is None:
            raise ValueError("incomplete inequality constraint (missing h)")
        if A is None and b is not None:
            raise ValueError("incomplete equality constraint (missing A)")
        if A is not None and b is None:
            raise ValueError("incomplete equality constraint (missing b)")
        # concatenate A with G and b with h for convenience and save the
        # first idx of the Lagrange multipliers constrained to be >= 0
        self.n_eq = self.A.shape[0] if self.A is not None else 0
        if self.A is not None and self.G is not None:
            self.AG = np.concatenate((self.A, self.G))
        elif self.A is not None:
            self.AG = self.A
        elif self.G is not None:
            self.AG = self.G
        else:
            self.AG = np.zeros((self.ndim, self.ndim))
        if self.b is not None and self.h is not None:
            self.bh = np.concatenate((self.b, self.h))
        elif self.b is not None:
            self.bh = self.b
        elif self.h is not None:
            self.bh = self.h
        else:
            self.bh = np.zeros(self.ndim)
        # initialize Lagrange multipliers to 0
        self.dual_x = np.zeros(self.AG.shape[0])

    def f_star(self):
        return self.primal.function(self.x_star())

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  A=self.A,
                                  b=self.b,
                                  G=self.G,
                                  h=self.h,
                                  solver='cvxopt')
        return self.x_opt

    def function(self, x):
        """
        Compute the value of the (possibly augmented) lagrangian relaxation defined as:

        L(x, mu, lambda) = 1/2 x^T Q x + q^T x + mu^T (A x - b) + lambda^T (G x - h) + rho/2 ||(A x - b) + (G x - h)||^2

        :param x: the primal variable wrt evaluate the function
        :return: the function value wrt primal-dual variable
        """
        violations = self.AG @ x - self.bh
        clipped_violations = violations.copy()
        clipped_violations[self.n_eq:] = np.clip(violations[self.n_eq:], a_min=0, a_max=None)
        return (super(LagrangianQuadratic, self).function(x) + self.dual_x @ violations +
                0.5 * self.rho * np.linalg.norm(clipped_violations) ** 2)

    def jacobian(self, x):
        """
        Compute the jacobian of the (possibly augmented) lagrangian relaxation defined as:

            J L(x, mu, lambda) = Q x + q + mu^T A + lambda^T G + rho ((A x - b) + (G x - h))

        :param x: the primal variable wrt evaluate the jacobian
        :return: the jacobian wrt primal-dual variable
        """
        violations = self.AG @ x - self.bh
        clipped_violations = violations.copy()
        clipped_violations[self.n_eq:] = np.clip(violations[self.n_eq:], a_min=0, a_max=None)
        violations = clipped_violations != 0
        return (super(LagrangianQuadratic, self).jacobian(x) + self.dual_x @ self.AG
                + self.rho * self.AG.T[:, violations] @ self.AG.T[:, violations].T @ x
                - self.rho * self.bh[violations] @ self.AG.T[:, violations].T)

    def hessian(self, x):
        return self.auto_hess(x)
