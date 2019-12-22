import autograd.numpy as np
from autograd import jacobian, hessian
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


class Function:
    def __init__(self):
        self._jacobian = jacobian(self.function)
        self._hessian = hessian(self.function)

    def function(self, x):
        return NotImplementedError

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of the function.
        :param x: 1-D array of points at which the Jacobian is to be computed.
        :return:  the Jacobian of the function at x.
        """
        return self._jacobian(np.array(x, dtype=float))

    def hessian(self, x):
        """
        The Hessian matrix of the function.
        :param x: 1-D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix of the function at x.
        """
        return self._hessian(np.array(x, dtype=float)).reshape((x.size, x.size))

    def plot(self, x_min, x_max, y_min, y_max):
        return NotImplementedError


class GenericQuadratic(Function):

    def __init__(self, Q, q):
        """
        Construct a general quadratic function with his linear and quadratic part.
        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (i.e. the quadratic part) of f. If it is not
                           positive semidefinite, f(x) will be unbounded below.
        :param q: ([n x 1] real column vector): the linear part of f.
        """
        Q = np.array(Q)
        q = np.array(q)

        # reading and checking input
        if not np.isrealobj(Q):
            raise ValueError('Q not a real matrix')

        n = Q.shape[0]

        if n <= 1:
            raise ValueError('Q is too small')

        if n != Q.shape[1]:
            raise ValueError('Q is not square')

        if not np.isrealobj(q):
            raise ValueError('q not a real vector')

        if q.shape[1] != 1:
            raise ValueError('q is not a (column) vector')

        if q.size != n:
            raise ValueError('q size does not match with Q')

        self.Q = Q
        self.q = q

    def function(self, x):
        """
        A general quadratic function f(x) = 1/2 x^T Q x - q^T x.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the value of a general quadratic function if x, the optimal solution of a
                  linear system Qx = q (=> x = Q^-1 q) which has a complexity of O(n^3) otherwise.
        """
        x = np.array(x)
        return (0.5 * x.T.dot(self.Q).dot(x) - self.q.T.dot(x)).item() \
            if x.size != 0 else self.function(np.linalg.inv(self.Q).dot(self.q)) \
            if min(np.linalg.eigvalsh(self.Q)) > 1e-14 else -np.inf  # np.linalg.solve(Q, q)

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of a general quadratic function J f(x) = Q x - q.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the Jacobian of a general quadratic function.
        """
        return self.Q.dot(x) - self.q  # complexity O(n^2)

    def hessian(self, x=None):
        """
        The Hessian matrix of a general quadratic function H f(x) = Q.
        :param x: 1-D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix (i.e. the quadratic part) of a general quadratic function at x.
        """
        return self.Q

    def plot(self, x_min=-5, x_max=1, y_min=-5, y_max=1):
        x, y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # 3D surface plot
        surface_plot = plt.figure()
        surface_axes = Axes3D(surface_plot)

        # generic quadratic function
        #                      T                           T
        # f(x, y) = 1/2 * | x |  * | a  b | * | x | - | d |  * | x |
        #                 | y |    | b  c |   | y |   | e |    | y |
        z = 0.5 * self.Q[0][0] * x ** 2 + self.Q[0][1] * x * y + \
            0.5 * self.Q[1][1] * y ** 2 - self.q[0] * x - self.q[1] * y

        surface_axes.plot_surface(x, y, z, norm=LogNorm(), cmap=cm.get_cmap('jet'))

        # 2D contour
        contour_plot, contour_axes = plt.subplots()

        contour_axes.contour(x, y, z, cmap=cm.get_cmap('jet'))
        contour_axes.plot(*np.linalg.inv(self.Q).dot(self.q), 'r*', markersize=10)  # np.linalg.solve(self.Q, self.q)

        return surface_plot, surface_axes, contour_plot, contour_axes


# generic 2x2 quadratic function with nicely conditioned Hessian
gen_quad_1 = GenericQuadratic([[6, -2], [-2, 6]], [[10], [5]])
# generic 2x2 quadratic function with less nicely conditioned Hessian
gen_quad_2 = GenericQuadratic([[5, -3], [-3, 5]], [[10], [5]])
# generic 2x2 quadratic function with Hessian having one zero eigenvalue
gen_quad_3 = GenericQuadratic([[4, -4], [-4, 4]], [[10], [5]])
# generic 2x2 quadratic function with indefinite Hessian
# (one positive and one negative eigenvalue)
gen_quad_4 = GenericQuadratic([[3, -5], [-5, 3]], [[10], [5]])
# generic 2x2 quadratic function with "very elongated" Hessian
# (a very small positive minimum eigenvalue, the other much larger)
gen_quad_5 = GenericQuadratic([[101, -99], [-99, 101]], [[10], [5]])


class Rosenbrock(Function):

    def function(self, x):
        """
        The Rosenbrock function.
        :param x: 1-D array of points at which the Rosenbrock function is to be computed.
        :return:  the value of the Rosenbrock function at x.
        """
        x = np.array(x)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2.0 + (1 - x[:-1]) ** 2) if x.size != 0 else 0

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
        contour_axes.plot(*np.array([1, 1]), 'r*', markersize=10)

        return surface_plot, surface_axes, contour_plot, contour_axes


class Ackley(Function):

    def function(self, x):
        """
        The Ackley function.
        :param x: 1-D array of points at which the Ackley function is to be computed.
        :return:  the value of the Ackley function.
        """
        x = np.array(x)
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / x.size)) \
               - np.exp((np.sum(np.cos(2.0 * np.pi * x))) / x.size) + np.e + 20 if x.size != 0 else 0

    def plot(self, x_min=-5, x_max=5, y_min=-5, y_max=5):
        x, y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # 3D surface plot
        surface_plot = plt.figure()
        surface_axes = Axes3D(surface_plot)

        # Ackley function
        z = -20 * np.exp(-0.2 * np.sqrt((x ** 2 + y ** 2) * 0.5)) \
            - np.exp((np.cos(2.0 * np.pi * x) + np.cos(2 * np.pi * y)) * 0.5) + np.e + 20

        surface_axes.plot_surface(x, y, z, norm=LogNorm(), cmap=cm.get_cmap('jet'))

        # 2D contour
        contour_plot, contour_axes = plt.subplots()

        contour_axes.contour(x, y, z, cmap=cm.get_cmap('jet'))
        contour_axes.plot(*np.array([0, 0]), 'r*', markersize=10)

        return surface_plot, surface_axes, contour_plot, contour_axes
