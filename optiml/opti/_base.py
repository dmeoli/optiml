from abc import ABC

import autograd.numpy as np
from autograd import jacobian, hessian
from scipy.linalg import cho_solve, cho_factor
from scipy.sparse.linalg import minres


class Optimizer(ABC):

    def __init__(self,
                 f,
                 x=None,
                 eps=1e-6,
                 tol=1e-8,
                 max_iter=1000,
                 callback=None,
                 callback_args=(),
                 random_state=None,
                 verbose=False):
        """

        :param f:        the objective function.
        :param x:        ([n x 1] real column vector): 1D array of points at which the Hessian is to be computed.
        :param eps:      (real scalar, optional, default value 1e-6): the accuracy in the stopping
                         criterion: the algorithm is stopped when the norm of the gradient is less
                         than or equal to eps.
        :param max_iter: (integer scalar, optional, default value 1000): the maximum number of iterations.
        :param verbose:  (boolean, optional, default value False): print details about each iteration
                         if True, nothing otherwise.
        """
        if not isinstance(f, OptimizationFunction):
            raise TypeError(f'{f} is not an allowed optimization function')
        self.f = f
        if x is None:
            if hasattr(self.f, 'primal'):  # is_lagrangian_dual()
                if hasattr(self.f, 'rho'):  # is_augmented_lagrangian_dual()
                    # dual_x is handled and initialized to 0 inside the `AugmentedLagrangianQuadratic`
                    # class, so initialize the primal variable, i.e., x, as a random uniform
                    if random_state is None:
                        x = np.random.uniform
                    else:
                        x = np.random.RandomState(random_state).uniform
                else:
                    # initialize the primal variable, i.e., x, as a random uniform and
                    # and the dual variable, i.e., mu_lmbda, to 0
                    if random_state is None:
                        x = np.concatenate((np.random.uniform(size=f.primal.ndim),  # x
                                            np.zeros(f.AG.shape[0])))  # mu_lmbda
                    else:
                        x = np.concatenate((np.random.RandomState(random_state).uniform(size=f.primal.ndim),  # x
                                            np.zeros(f.AG.shape[0])))  # mu_lmbda
            else:
                if random_state is None:
                    x = np.random.uniform
                else:
                    x = np.random.RandomState(random_state).uniform
        if callable(x):
            try:
                self.x = x(size=f.ndim)
            except TypeError:
                self.x = x(shape=f.ndim)
        else:
            self.x = np.asarray(x, dtype=float)
        self.f_x = np.nan
        if self.is_lagrangian_dual():
            self.past_x = self.x.copy()
            self.primal_f_x = np.nan
            self.dgap = np.nan
        self.g_x = np.zeros(0)
        self.eps = eps
        self.tol = tol
        if not max_iter > 0:
            raise ValueError('max_iter must be > 0')
        self.max_iter = max_iter
        self.iter = 0
        self.status = 'unknown'
        if (self.f.ndim <= 3 or
                hasattr(self.f, 'primal') and self.f.primal.ndim <= 3):
            self.x0_history = []
            self.x1_history = []
            self.f_x_history = []
        self._callback = callback
        self.callback_args = callback_args
        self.random_state = random_state
        self.verbose = verbose

    def is_lagrangian_dual(self):
        return hasattr(self.f, 'primal')

    def is_augmented_lagrangian_dual(self):
        return self.is_lagrangian_dual() and hasattr(self.f, 'rho')

    def callback(self, args=()):

        if hasattr(self.f, 'primal'):  # is_lagrangian_dual()

            if hasattr(self.f, 'rho'):  # is_augmented_lagrangian_dual()
                self.primal_f_x = self.f.primal.function(self.x)
            else:
                self.primal_f_x = self.f.primal.function(self.x[:self.f.primal.ndim])

            self.dgap = abs((self.primal_f_x - self.f_x) / max(abs(self.primal_f_x), 1))

            if self.is_verbose():
                print('\tpcost: {: 1.4e}'.format(self.primal_f_x), end='')
                print('\tdgap: {: 1.4e}'.format(self.dgap), end='')

            if self.f.primal.ndim == 2:
                self.x0_history.append(self.x[0])
                self.x1_history.append(self.x[1])
                self.f_x_history.append(self.primal_f_x)

            if callable(self._callback):  # custom callback
                self._callback(self, *args, *self.callback_args)

            self.past_x = self.x.copy()  # backup primal x before upgrade it outside the callback

        else:

            if self.f.ndim <= 3:
                self.x0_history.append(self.x[0])
                self.x1_history.append(self.x[1])
                self.f_x_history.append(self.f_x)

            if callable(self._callback):  # custom callback
                self._callback(self, *args, *self.callback_args)

    def check_lagrangian_dual_optimality(self):

        if hasattr(self.f, 'primal'):  # is_lagrangian_dual()

            constraints = self.f.constraints(self.x)

            if hasattr(self.f, 'rho'):  # is_augmented_lagrangian_dual()

                self.f.past_dual_x = self.f.dual_x.copy()  # backup dual_x before upgrade it

                # upgrade dual_x and clip lmbda
                self.f.dual_x += self.f.rho * constraints
                self.f.dual_x[self.f.n_eq:] = np.clip(self.f.dual_x[self.f.n_eq:], a_min=0, a_max=None)

                # check optimality conditions
                if ((np.linalg.norm(self.f.dual_x - self.f.past_dual_x) +
                     np.linalg.norm(self.x - self.past_x) <= self.tol) or
                        np.linalg.norm(constraints) <= self.tol):
                    self.status = 'optimal'
                    raise StopIteration

            else:

                # clip lmbda and backup mu_lmbda
                self.x[self.f.primal.ndim + self.f.n_eq:] = np.clip(self.x[self.f.primal.ndim + self.f.n_eq:],
                                                                    a_min=0, a_max=None)
                self.f.dual_x = self.x[self.f.primal.ndim:].copy()

                # check optimality conditions
                if ((np.linalg.norm(self.x - self.past_x) <= self.tol) or  # x_mu_lmbda - past_x_mu_lmbda
                        np.linalg.norm(constraints) <= self.tol):
                    self.status = 'optimal'
                    raise StopIteration

    def check_lagrangian_dual_conditions(self):
        # check if the Lagrange multipliers that controls the inequality constraints are >= 0
        if hasattr(self.f, 'primal'):  # is_lagrangian_dual()
            if hasattr(self.f, 'rho'):  # is_augmented_lagrangian_dual()
                assert all(self.f.dual_x[self.f.n_eq:] >= 0)
            else:
                assert all(self.x[self.f.primal.ndim + self.f.n_eq:] >= 0)

    def is_verbose(self):
        return self.verbose and not self.iter % self.verbose

    def minimize(self):
        raise NotImplementedError

    def _print_header(self):
        raise NotImplementedError

    def _print_info(self):
        raise NotImplementedError


class OptimizationFunction(ABC):

    def __init__(self, ndim=2):
        self.auto_jac = jacobian(self.function)
        self.auto_hess = hessian(self.function)
        self.ndim = ndim

    def x_star(self):
        return np.full(fill_value=np.nan, shape=self.ndim)

    def f_star(self):
        return np.inf

    def args(self):
        return ()

    def function(self, x):
        raise NotImplementedError

    def jacobian(self, x):
        """
        The Jacobian (i.e., the gradient) of the function.
        :param x: 1D array of points at which the Jacobian is to be computed.
        :return:  the Jacobian of the function at x.
        """
        return self.auto_jac(x)

    def function_jacobian(self, *args, **kwargs):
        return self.function(*args, **kwargs), self.jacobian(*args, **kwargs)

    def hessian(self, x):
        """
        The Hessian matrix of the function.
        :param x: 1D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix of the function at x.
        """
        return self.auto_hess(x)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class Quadratic(OptimizationFunction):

    def __init__(self, Q, q):
        """
        Construct a quadratic function from its linear and quadratic part defined as:

                                    1/2 x^T Q x + q^T x

        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (i.e., the quadratic part) of f. If it is not
                           positive semidefinite, f(x) will be unbounded below.
        :param q: ([n x 1] real column vector): the linear part of f.
        """
        Q = np.array(Q)
        q = np.array(q)

        n = len(Q)
        super(Quadratic, self).__init__(n)

        if n <= 1:
            raise ValueError('Q is too small')
        if n != Q.shape[0]:
            raise ValueError('Q is not square')
        self.Q = Q

        if q.size != n:
            raise ValueError('q size does not match with Q')
        self.q = q

    def x_star(self):
        if not hasattr(self, 'x_opt'):
            try:
                # use the Cholesky factorization to solve the linear system if Q is
                # symmetric and positive definite, i.e., the function is strictly convex
                self.x_opt = cho_solve(cho_factor(self.Q), -self.q)
            except np.linalg.LinAlgError:
                # since Q is is not strictly psd, i.e., the function is linear along the
                # eigenvectors correspondent to the null eigenvalues, the system has infinite
                # solutions, so we will choose the one that minimizes the residue
                self.x_opt = minres(self.Q, -self.q)[0]
        return self.x_opt

    def f_star(self):
        return self.function(self.x_star())

    def function(self, x):
        """
        A general quadratic function f(x) = 1/2 x^T Q x + q^T x.
        :param x: ([n x 1] real column vector): 1D array of points at which the Hessian is to be computed.
        :return:  the value of a general quadratic function if x, the optimal solution of a
                  linear system Qx = q (=> x = Q^-1 q) which has a complexity of O(n^3) otherwise.
        """
        return 0.5 * x @ self.Q @ x + self.q @ x

    def jacobian(self, x):
        """
        The Jacobian (i.e., the gradient) of a general quadratic function J f(x) = Q x + q.
        :param x: ([n x 1] real column vector): 1D array of points at which the Hessian is to be computed.
        :return:  the Jacobian of a general quadratic function.
        """
        return self.Q @ x + self.q

    def hessian(self, x):
        """
        The Hessian matrix of a general quadratic function H f(x) = Q.
        :param x: 1D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix (i.e., the the quadratic part) of a general quadratic function at x.
        """
        return self.Q


# 2x2 quadratic function with nicely conditioned Hessian
quad1 = Quadratic(Q=[[6, -2], [-2, 6]], q=[10, 5])
# 2x2 quadratic function with less nicely conditioned Hessian
quad2 = Quadratic(Q=[[5, -3], [-3, 5]], q=[10, 5])
# 2x2 quadratic function with Hessian having one zero eigenvalue (singular matrix)
quad3 = Quadratic(Q=[[4, -4], [-4, 4]], q=[10, 5])
# 2x2 quadratic function with indefinite Hessian (one positive and one negative eigenvalue)
quad4 = Quadratic(Q=[[3, -5], [-5, 3]], q=[10, 5])
# 2x2 quadratic function with "very elongated" Hessian
# (a very small positive minimum eigenvalue, the other much larger)
quad5 = Quadratic(Q=[[101, -99], [-99, 101]], q=[10, 5])
