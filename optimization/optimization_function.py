import autograd.numpy as np
from autograd import jacobian, hessian
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


class OptimizationFunction:
    def __init__(self, n=2):
        self.jac = jacobian(self.function)
        self.hes = hessian(self.function)
        self.x_opt = None
        self.n = n

    def x_star(self):
        return self.x_opt

    def f_star(self):
        return self.function(self.x_star(), *self.args())

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
        return self.jac(np.asarray(x, dtype=float))

    def hessian(self, x):
        """
        The Hessian matrix of the function.
        :param x: 1D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix of the function at x.
        """
        return self.hes(np.asarray(x, dtype=float)).reshape((x.size, x.size))

    def plot(self, x_min, x_max, y_min, y_max):
        raise NotImplementedError


class Quadratic(OptimizationFunction):

    def __init__(self, Q, q):
        """
        Construct a general quadratic function with his linear and quadratic part.
        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (i.e. the quadratic part) of f. If it is not
                           positive semidefinite, f(x) will be unbounded below.
        :param q: ([n x 1] real column vector): the linear part of f.
        """
        super().__init__(Q.shape[1])
        self.Q = Q
        self.q = q

    def x_star(self):
        if self.x_opt is not None:
            return self.x_opt
        else:
            try:
                self.x_opt = np.linalg.inv(self.Q).dot(self.q)  # or np.linalg.solve(self.Q, self.q)
            except np.linalg.LinAlgError:
                self.x_opt = np.full((self.n,), np.nan)
            return self.x_opt

    def args(self):
        return self.Q, self.q

    def function(self, x, Q, q):
        """
        A general quadratic function f(x) = 1/2 x^T Q x - q^T x.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the value of a general quadratic function if x, the optimal solution of a
                  linear system Qx = q (=> x = Q^-1 q) which has a complexity of O(n^3) otherwise.
        """
        return 0.5 * x.T.dot(Q).dot(x) - q.T.dot(x)

    def jacobian(self, x, Q, q):
        """
        The Jacobian (i.e. gradient) of a general quadratic function J f(x) = Q x - q.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the Jacobian of a general quadratic function.
        """
        return Q.dot(x) - q

    def hessian(self, x, Q, q):
        """
        The Hessian matrix of a general quadratic function H f(x) = Q.
        :param x: 1D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix (i.e. the quadratic part) of a general quadratic function at x.
        """
        return Q

    def plot(self, x_min=-5, x_max=2, y_min=-5, y_max=2):
        x, y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # 3D surface plot
        surface_plot = plt.figure()
        surface_axes = Axes3D(surface_plot)

        #                      T                           T
        # f(x, y) = 1/2 * | x |  * | a  b | * | x | - | d |  * | x |
        #                 | y |    | b  c |   | y |   | e |    | y |
        z = 0.5 * self.Q[0][0] * x ** 2 + self.Q[0][1] * x * y + \
            0.5 * self.Q[1][1] * y ** 2 - self.q[0] * x - self.q[1] * y

        surface_axes.plot_surface(x, y, z, norm=LogNorm(), cmap=cm.get_cmap('jet'))

        # 2D contour
        contour_plot, contour_axes = plt.subplots()

        contour_axes.contour(x, y, z, cmap=cm.get_cmap('jet'))
        contour_axes.plot(*self.x_star(), 'r*', markersize=10)
        return surface_plot, surface_axes, contour_plot, contour_axes


# 2x2 quadratic function with nicely conditioned Hessian
quad1 = Quadratic(np.array([[6, -2], [-2, 6]]), np.array([10, 5]))
# 2x2 quadratic function with less nicely conditioned Hessian
quad2 = Quadratic(np.array([[5, -3], [-3, 5]]), np.array([10, 5]))
# 2x2 quadratic function with Hessian having one zero eigenvalue
# (singular matrix)
quad3 = Quadratic(np.array([[4, -4], [-4, 4]]), np.array([10, 5]))
# 2x2 quadratic function with indefinite Hessian
# (one positive and one negative eigenvalue)
quad4 = Quadratic(np.array([[3, -5], [-5, 3]]), np.array([10, 5]))
# 2x2 quadratic function with "very elongated" Hessian
# (a very small positive minimum eigenvalue, the other much larger)
quad5 = Quadratic(np.array([[101, -99], [-99, 101]]), np.array([10, 5]))


class Rosenbrock(OptimizationFunction):

    def __init__(self, n=2, a=1, b=2):
        super().__init__(n)
        self.a = a
        self.b = b

    def x_star(self):
        if self.x_opt is not None:
            return self.x_opt
        else:
            # only in the trivial case where a = 0 the function is symmetric and the minimum is at the origin
            self.x_opt = np.zeros(self.n) if self.a is 0 else np.ones(self.n)
            return self.x_opt

    def function(self, x):
        """
        The Rosenbrock function.
        :param x: 1D array of points at which the Rosenbrock function is to be computed.
        :return:  the value of the Rosenbrock function at x.
        """
        return np.sum(self.b * (x[1:] - x[:-1] ** 2) ** 2 + (self.a - x[:-1]) ** 2)

    def plot(self, x_min=-2, x_max=2, y_min=-1, y_max=3):
        x, y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # 3D surface plot
        surface_plot = plt.figure()
        surface_axes = Axes3D(surface_plot)

        # Rosenbrock function
        z = 100. * (y - x ** 2) ** 2 + (1. - x) ** 2

        surface_axes.plot_surface(x, y, z, norm=LogNorm(), cmap=cm.get_cmap('jet'))

        # 2D contour
        contour_plot, contour_axes = plt.subplots()

        contour_axes.contour(x, y, z, cmap=cm.get_cmap('jet'))
        contour_axes.plot(*self.x_star(), 'r*', markersize=10)

        return surface_plot, surface_axes, contour_plot, contour_axes
