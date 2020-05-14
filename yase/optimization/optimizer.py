import autograd.numpy as np
from autograd import jacobian, hessian
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm


class Optimizer:

    def __init__(self, f, x, eps=1e-6, max_iter=1000,
                 callback=None, callback_args=(), verbose=False):
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
            raise TypeError('f is not an optimization function')
        self.f = f
        if callable(x):
            self.x = x(f.ndim)
        elif not np.isrealobj(x):
            raise ValueError('x not a real vector')
        else:
            self.x = np.asarray(x, dtype=np.float)
        self.f_x = np.nan
        self.g_x = np.zeros(0)
        if self.f.ndim == 2:
            self.x0_history = []
            self.x1_history = []
            self.f_x_history = []
        if not np.isscalar(eps):
            raise ValueError('eps is not a real scalar')
        if not eps > 0:
            raise ValueError('eps must be > 0')
        self.eps = eps
        if not np.isscalar(max_iter):
            raise ValueError('max_iter is not an integer scalar')
        if not max_iter > 0:
            raise ValueError('max_iter must be > 0')
        self.max_iter = max_iter
        self.iter = 0
        self._callback = callback
        self.callback_args = callback_args
        self.status = 'unknown'
        self.verbose = verbose

    def callback(self, args=None):
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

    def __init__(self, ndim):
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

    def plot(self, x_min, x_max, y_min, y_max):
        X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = np.array([self.function(np.array([x, y]))
                      for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

        fig = plt.figure(figsize=(16, 8))

        # 3D surface plot
        ax = fig.add_subplot(1, 2, 1, projection='3d', elev=50, azim=-50)
        ax.plot_surface(X, Y, Z, norm=SymLogNorm(linthresh=abs(Z.min()), base=np.e), cmap='jet', alpha=0.5)
        ax.plot([self.x_star()[0]], [self.x_star()[1]], [self.f_star()], marker='*', color='r', markersize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(f'${type(self).__name__}$')

        # 2D contour plot
        ax = fig.add_subplot(1, 2, 2)
        ax.contour(X, Y, Z, 70, cmap='jet')
        ax.plot(*self.x_star(), marker='*', color='r', markersize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        return fig


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

        if not np.isrealobj(Q):
            raise ValueError('Q not a real matrix')

        n = Q.shape[1]
        super().__init__(n)

        if n <= 1:
            raise ValueError('Q is too small')
        if n != Q.shape[0]:
            raise ValueError('Q is not square')
        self.Q = Q

        if not np.isrealobj(q):
            raise ValueError('q not a real vector')
        if q.size != n:
            raise ValueError('q size does not match with Q')
        self.q = q

    def x_star(self):
        try:
            return np.linalg.solve(self.Q, -self.q)  # complexity O(2n^3/3)
        except np.linalg.LinAlgError:  # the Hessian matrix is singular
            return super().x_star()

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


class Rosenbrock(OptimizationFunction):

    def __init__(self, ndim=2, a=1, b=2):
        super().__init__(ndim)
        self.a = a
        self.b = b

    def x_star(self):
        return np.zeros(self.ndim) if self.a == 0 else np.ones(self.ndim)

    def f_star(self):
        return self.function(self.x_star())

    def function(self, x):
        """
        The Rosenbrock function.
        :param x: 1D array of points at which the Rosenbrock function is to be computed.
        :return:  the value of the Rosenbrock function at x.
        """
        return np.sum(self.b * (x[1:] - x[:-1] ** 2) ** 2 + (self.a - x[:-1]) ** 2)
