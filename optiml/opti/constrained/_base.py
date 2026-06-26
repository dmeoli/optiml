from abc import ABC

import autograd.numpy as np
from autograd import hessian, jacobian
from qpsolvers import solve_qp

from optiml.opti import Optimizer, Quadratic


class BoxConstrainedQuadraticOptimizer(Optimizer, ABC):
    r"""
    Abstract base class for the optimizers that solve the convex Box-Constrained
    Quadratic program

    .. math::

        (P) \quad \min \left\{ \tfrac{1}{2} x^\top Q x + q^\top x : 0 \le x \le ub \right\}
    """

    def __init__(self,
                 quad,
                 ub,
                 lb=None,
                 x=None,
                 eps=1e-6,
                 tol=1e-8,
                 max_iter=1000,
                 callback=None,
                 callback_args=(),
                 verbose=False):
        r"""

        :param quad:          the quadratic function :math:`\tfrac{1}{2} x^\top Q x + q^\top x` to be minimized.
        :param ub:            ([n x 1] real column vector): the upper bound of the box, i.e., the
                              variables are constrained to lie in :math:`lb \le x \le ub`.
        :param lb:            ([n x 1] real column vector, optional): the lower bound of the box; if not
                              provided it defaults to the all-zeros vector, i.e., :math:`0 \le x \le ub`.
        :param x:             ([n x 1] real column vector, optional): the point where to start the
                              algorithm from; if not provided, it starts from the middle of the box.
        :param eps:           (real scalar, optional, default value 1e-6): the accuracy in the stopping
                              criterion: the algorithm is stopped when the norm of the gradient is less
                              than or equal to eps.
        :param tol:           (real scalar, optional, default value 1e-8): the tolerance used to check the
                              optimality conditions when f is a Lagrangian dual relaxation.
        :param max_iter:      (integer scalar, optional, default value 1000): the maximum number of iterations.
        :param callback:      (callable, optional, default value None): a function called at each iteration
                              with the optimizer instance as first argument.
        :param callback_args: (tuple, optional, default value ()): additional arguments passed to callback.
        :param verbose:       (boolean, optional, default value False): print details about each iteration
                              if True, nothing otherwise.
        :return x:            ([n x 1] real column vector): the best solution found so far.
        :return status: (string): the status of the algorithm at termination, one of: ``optimal`` (x is a(n approximately) optimal solution); ``unbounded`` (f() was driven below m_inf, i.e., the problem looks unbounded below); ``stopped`` (the maximum number of iterations/evaluations was reached); ``error`` (a numerical error occurred, e.g., the step size fell below min_a).
        """
        if not isinstance(quad, Quadratic):
            raise TypeError(f'{quad} is not an allowed quadratic function')
        ub = np.asarray(ub, dtype=float)
        lb = np.zeros_like(ub) if lb is None else np.asarray(lb, dtype=float)
        super(BoxConstrainedQuadraticOptimizer, self).__init__(f=quad,
                                                               # starts from the middle of the box
                                                               x=x if x is not None else (lb + ub) / 2,
                                                               eps=eps,
                                                               tol=tol,
                                                               max_iter=max_iter,
                                                               callback=callback,
                                                               callback_args=callback_args,
                                                               verbose=verbose)
        self.lb = lb
        self.ub = ub

    def f_star(self):
        return self.f.function(self.x_star())

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.f.Q,
                                  q=self.f.q,
                                  lb=self.lb,
                                  ub=self.ub,
                                  solver='quadprog')
        return self.x_opt


class LagrangianQuadratic(Quadratic):
    r"""
    Construct the lagrangian relaxation of a constrained quadratic function defined as

    .. math::

        \tfrac{1}{2} x^\top Q x + q^\top x : A x = b, \; G x \le h, \; lb \le x \le ub

    i.e.,

    .. math::

        \tfrac{1}{2} x^\top Q x + q^\top x : A x = b, \; \hat{G} x \le \hat{h}

    where :math:`\hat{G}^\top = [\, G \;\; -I \;\; I \,]` and :math:`\hat{h} = [\, h \;\; -lb \;\; ub \,]`.
    """

    def __init__(self, primal, A=None, b=None, G=None, h=None, lb=None, ub=None):
        if not isinstance(primal, Quadratic):
            raise TypeError(f'{primal} is not an allowed quadratic function')
        super(LagrangianQuadratic, self).__init__(primal.Q, primal.q)
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
        if G is None and h is not None:
            raise ValueError('incomplete inequality constraint (missing G)')
        if G is not None and h is None:
            raise ValueError('incomplete inequality constraint (missing h)')
        if A is None and b is not None:
            raise ValueError('incomplete equality constraint (missing A)')
        if A is not None and b is None:
            raise ValueError('incomplete equality constraint (missing b)')
        # concatenate A with G and b with h for convenience and save the
        # first idx of the Lagrange multipliers constrained to be >= 0
        self.n_eq = self.A.shape[0] if self.A is not None else 0
        if self.A is not None and self.G is not None:
            self.AG = np.concatenate((self.A, self.G))
        elif self.A is not None:
            self.AG = self.A
            self.G = np.zeros((self.ndim, self.ndim))  # G is None
        elif self.G is not None:
            self.AG = self.G
            self.A = np.zeros((self.ndim, self.ndim))  # A is None
        else:
            self.A = np.zeros((self.ndim, self.ndim))
            self.G = np.zeros((self.ndim, self.ndim))
            self.AG = np.concatenate((self.A, self.G))  # A and G are None
        if self.b is not None and self.h is not None:
            self.bh = np.concatenate((self.b, self.h))
        elif self.b is not None:
            self.h = np.zeros(self.ndim)  # h is None
            self.bh = self.b
        elif self.h is not None:
            self.b = np.zeros(self.ndim)  # b is None
            self.bh = self.h
        else:
            self.b = np.zeros(self.ndim)
            self.h = np.zeros(self.ndim)
            self.bh = np.concatenate((self.b, self.h))  # b and h are None
        self.ndim += self.AG.shape[0]
        # backup Lagrange multipliers
        self.dual_x = None  # mu_lmbda

    def f_star(self):
        return self.primal.function(self.x_star())

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  A=self.A if not np.all((self.A == 0)) else None,
                                  b=self.b if not np.all((self.A == 0)) else None,  # check for A since b can be zero
                                  G=self.G if not np.all((self.G == 0)) else None,
                                  h=self.h if not np.all((self.G == 0)) else None,  # check for G since h can be zero
                                  solver='cvxopt')
        return self.x_opt

    def constraints(self, x_mu_lmbda):
        return self.AG @ x_mu_lmbda[:self.primal.ndim] - self.bh

    def function(self, x_mu_lmbda):
        r"""
        Compute the value of the augmented lagrangian relaxation defined as

        .. math::

            L(x, \mu, \lambda) = \tfrac{1}{2} x^\top Q x + q^\top x + \mu^\top (A x - b) + \lambda^\top (G x - h)

        :param x_mu_lmbda: the primal-dual variable wrt evaluate the function
        :return: the function value wrt primal-dual variable
        """
        x, mu_lmbda = np.split(x_mu_lmbda, [self.primal.ndim])
        return self.primal.function(x) + mu_lmbda @ (self.AG @ x - self.bh)

    def jacobian(self, x_mu_lmbda):
        r"""
        Compute the jacobian of the lagrangian relaxation defined as

        .. math::

            J L(x, \mu, \lambda) = Q x + q + \mu^\top A + \lambda^\top G

        :param x_mu_lmbda: the primal-dual variable wrt evaluate the jacobian
        :return: the jacobian wrt primal-dual variable
        """
        # jac = self.auto_jac(x_mu_lmbda)  # slower
        x, mu_lmbda = np.split(x_mu_lmbda, [self.primal.ndim])
        jac = np.concatenate((self.primal.jacobian(x) + mu_lmbda @ self.AG,  # gradient wrt x
                              self.A @ x - self.b if not np.all((self.A == 0)) else [],  # gradient wrt mu
                              self.G @ x - self.h if not np.all((self.G == 0)) else []))  # gradient wrt lmbda
        # gradient ascent for the dual since we need to maximize wrt mu_lmbda, so we change the sign
        jac[self.primal.ndim:] = -jac[self.primal.ndim:]
        return jac

    def hessian(self, x):
        return self.auto_hess(x)


class AugmentedLagrangianQuadratic(Quadratic):
    r"""
    Construct the augmented lagrangian relaxation of a constrained quadratic function defined as

    .. math::

        \tfrac{1}{2} x^\top Q x + q^\top x : A x = b, \; G x \le h, \; lb \le x \le ub

    i.e.,

    .. math::

        \tfrac{1}{2} x^\top Q x + q^\top x : A x = b, \; \hat{G} x \le \hat{h}

    where :math:`\hat{G}^\top = [\, G \;\; -I \;\; I \,]` and :math:`\hat{h} = [\, h \;\; -lb \;\; ub \,]`.
    """

    def __init__(self, primal, A=None, b=None, G=None, h=None, lb=None, ub=None, rho=1):
        if not isinstance(primal, Quadratic):
            raise TypeError(f'{primal} is not an allowed quadratic function')
        super(AugmentedLagrangianQuadratic, self).__init__(primal.Q, primal.q)
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
        if G is None and h is not None:
            raise ValueError('incomplete inequality constraint (missing G)')
        if G is not None and h is None:
            raise ValueError('incomplete inequality constraint (missing h)')
        if A is None and b is not None:
            raise ValueError('incomplete equality constraint (missing A)')
        if A is not None and b is None:
            raise ValueError('incomplete equality constraint (missing b)')
        if not rho > 0:
            raise ValueError('rho must be must > 0')
        self.rho = rho
        # concatenate A with G and b with h for convenience and save the
        # first idx of the Lagrange multipliers constrained to be >= 0
        self.n_eq = self.A.shape[0] if self.A is not None else 0
        if self.A is not None and self.G is not None:
            self.AG = np.concatenate((self.A, self.G))
        elif self.A is not None:
            self.AG = self.A
            self.G = np.zeros((self.ndim, self.ndim))  # G is None
        elif self.G is not None:
            self.AG = self.G
            self.A = np.zeros((self.ndim, self.ndim))  # A is None
        else:
            self.A = np.zeros((self.ndim, self.ndim))
            self.G = np.zeros((self.ndim, self.ndim))
            self.AG = np.concatenate((self.A, self.G))  # A and G are None
        if self.b is not None and self.h is not None:
            self.bh = np.concatenate((self.b, self.h))
        elif self.b is not None:
            self.h = np.zeros(self.ndim)  # h is None
            self.bh = self.b
        elif self.h is not None:
            self.b = np.zeros(self.ndim)  # b is None
            self.bh = self.h
        else:
            self.b = np.zeros(self.ndim)
            self.h = np.zeros(self.ndim)
            self.bh = np.concatenate((self.b, self.h))  # b and h are None
        # initialize Lagrange multipliers to 0
        self.dual_x = np.zeros(self.AG.shape[0])  # mu_lmbda
        self.past_dual_x = self.dual_x.copy()
        # overwrite autograd utils
        self.auto_jac = jacobian(self._autograd_function)
        self.auto_hess = hessian(self._autograd_function)
        # backup {x: constraints} to speedup by reducing
        # the number of matrix-vector products
        self.last_x = None
        self.last_constraints = None

    def f_star(self):
        return self.primal.function(self.x_star())

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            self.x_opt = solve_qp(P=self.Q,
                                  q=self.q,
                                  A=self.A if not np.all((self.A == 0)) else None,
                                  b=self.b if not np.all((self.A == 0)) else None,  # check for A since b can be zero
                                  G=self.G if not np.all((self.G == 0)) else None,
                                  h=self.h if not np.all((self.G == 0)) else None,  # check for G since h can be zero
                                  solver='cvxopt')
        return self.x_opt

    def constraints(self, x):
        if np.array_equal(self.last_x, x):
            constraints = self.last_constraints.copy()  # speedup: just restore
        else:
            constraints = self.AG @ x - self.bh
            # backup {x: constraints}
            self.last_x = x.copy()
            self.last_constraints = constraints.copy()
        return constraints

    def function(self, x):
        r"""
        Compute the value of the augmented lagrangian relaxation defined as

        .. math::

            L(x, \mu, \lambda) = \tfrac{1}{2} x^\top Q x + q^\top x + \mu^\top (A x - b) + \lambda^\top (G x - h) + \tfrac{\rho}{2} \| (A x - b) + (G x - h) \|^2

        :param x: the primal variable wrt evaluate the function
        :return: the function value wrt primal-dual variable
        """
        constraints = self.constraints(x)
        clipped_constraints = constraints.copy()
        clipped_constraints[self.n_eq:] = np.clip(constraints[self.n_eq:], a_min=0, a_max=None)
        return (self.primal.function(x) + self.dual_x @ constraints +
                0.5 * self.rho * np.linalg.norm(clipped_constraints) ** 2)

    def _autograd_function(self, x):
        r"""
        Compute the value of the augmented lagrangian relaxation defined as

        .. math::

            L(x, \mu, \lambda) = \tfrac{1}{2} x^\top Q x + q^\top x + \mu^\top (A x - b) + \lambda^\top (G x - h) + \tfrac{\rho}{2} \| (A x - b) + (G x - h) \|^2

        Returns the same value of `function(self, x)` but it is written avoiding vector assignments
        to make it understandable by autograd, so it perform more matrix-vector products and for this
        reason it is more computationally expensive.

        :param x: the primal variable wrt evaluate the function
        :return: the function value wrt primal-dual variable
        """
        return (self.primal.function(x) + self.dual_x @ (self.AG @ x - self.bh) +
                0.5 * self.rho * np.sum(np.square(self.A @ x - self.b)) +
                0.5 * self.rho * np.sum(np.square(np.clip(self.G @ x - self.h, a_min=0, a_max=None))))

    def jacobian(self, x):
        r"""
        Compute the jacobian of the augmented lagrangian relaxation defined as

        .. math::

            J L(x, \mu, \lambda) = Q x + q + \mu^\top A + \lambda^\top G + \rho ((A x - b) + (G x - h))

        :param x: the primal variable wrt evaluate the jacobian
        :return: the jacobian wrt primal-dual variable
        """
        # return self.auto_jac(x)  # slower
        constraints = self.constraints(x)
        clipped_constraints = constraints.copy()
        clipped_constraints[self.n_eq:] = np.clip(constraints[self.n_eq:], a_min=0, a_max=None)
        idx_nonclipped = clipped_constraints != 0
        return (self.primal.jacobian(x) + self.dual_x @ self.AG +
                self.rho * self.AG[idx_nonclipped].T @ self.AG[idx_nonclipped] @ x -
                self.rho * self.bh[idx_nonclipped] @ self.AG[idx_nonclipped])

    def function_jacobian(self, x):
        constraints = self.constraints(x)
        clipped_constraints = constraints.copy()
        clipped_constraints[self.n_eq:] = np.clip(constraints[self.n_eq:], a_min=0, a_max=None)
        fun = (self.primal.function(x) + self.dual_x @ constraints +
               0.5 * self.rho * np.linalg.norm(clipped_constraints) ** 2)
        idx_nonclipped = clipped_constraints != 0
        jac = (self.primal.jacobian(x) + self.dual_x @ self.AG +
               self.rho * self.AG[idx_nonclipped].T @ self.AG[idx_nonclipped] @ x -
               self.rho * self.bh[idx_nonclipped] @ self.AG[idx_nonclipped])
        return fun, jac

    def hessian(self, x):
        return self.auto_hess(x)
