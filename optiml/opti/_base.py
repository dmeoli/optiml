import autograd.numpy as np
from autograd import jacobian, hessian


class Optimizer:

    def __init__(self, f, x, eps=1e-6, max_iter=1000, callback=None, callback_args=(), verbose=False):
        """

        :param f:        the objective function.
        :param x:        ([n x 1] real column vector): the point where to start the algorithm from.
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
        if callable(x):
            self.x = x(f.ndim)
        else:
            self.x = np.asarray(x, dtype=np.float)
        self.f_x = np.nan
        self.g_x = np.zeros(0)
        if self.f.ndim == 2:
            self.x0_history = []
            self.x1_history = []
            self.f_x_history = []
        if not eps > 0:
            raise ValueError('eps must be > 0')
        self.eps = eps
        if not max_iter > 0:
            raise ValueError('max_iter must be > 0')
        self.max_iter = max_iter
        self.iter = 0
        self._callback = callback
        self.callback_args = callback_args
        self.status = 'unknown'
        self.verbose = verbose

    def callback(self, args=()):
        if self.f.ndim == 2:
            self.x0_history.append(self.x[0])
            self.x1_history.append(self.x[1])
            self.f_x_history.append(self.f_x)
        if callable(self._callback):
            self._callback(self, *args, *self.callback_args)

    def is_verbose(self):
        return self.verbose and not self.iter % self.verbose

    def minimize(self):
        raise NotImplementedError


class OptimizationFunction:

    def __init__(self, ndim=2):
        self.jac = jacobian(self.function)
        self.hes = hessian(self.function)
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
        The Jacobian (i.e. gradient) of the function.
        :param x: 1D array of points at which the Jacobian is to be computed.
        :return:  the Jacobian of the function at x.
        """
        return self.jac(x)

    def hessian(self, x):
        """
        The Hessian matrix of the function.
        :param x: 1D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix of the function at x.
        """
        return self.hes(x)


class Quadratic(OptimizationFunction):

    def __init__(self, Q, q):
        """
        Construct a quadratic function from its linear and quadratic part defined as:

                                    1/2 x^T Q x + q^T x

        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (i.e. the quadratic part) of f. If it is not
                           positive semidefinite, f(x) will be unbounded below.
        :param q: ([n x 1] real column vector): the linear part of f.
        """
        Q = np.array(Q)
        q = np.array(q)

        n = len(Q)
        super().__init__(n)

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
                self.x_opt = np.linalg.solve(self.Q, -self.q)  # complexity O(2n^3/3)
            except np.linalg.LinAlgError:  # the Hessian matrix is singular
                self.x_opt = np.full(fill_value=np.nan, shape=self.ndim)
        return self.x_opt

    def f_star(self):
        return self.function(self.x_star())

    def function(self, x):
        """
        A general quadratic function f(x) = 1/2 x^T Q x + q^T x.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the value of a general quadratic function if x, the optimal solution of a
                  linear system Qx = q (=> x = Q^-1 q) which has a complexity of O(n^3) otherwise.
        """
        return 0.5 * x.T.dot(self.Q).dot(x) + self.q.T.dot(x)

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of a general quadratic function J f(x) = Q x + q.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the Jacobian of a general quadratic function.
        """
        return self.Q.dot(x) + self.q

    def hessian(self, x):
        """
        The Hessian matrix of a general quadratic function H f(x) = Q.
        :param x: 1D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix (i.e. the quadratic part) of a general quadratic function at x.
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
