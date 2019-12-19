import autograd.numpy as np
from autograd import jacobian, hessian
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


class GenericQuadratic:

    def __init__(self, Q, q):
        """

        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (i.e. the quadratic part) of f. If it is not
                           positive semidefinite, f(x) will be unbounded below
        :param q: ([n x 1] real column vector): the linear part of f
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
        A general quadratic function f(x) = 1/2 x^T Q x + q^T x.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from
        :return: the value of a general quadratic function if x, the optimal solution of a
        linear system Qx = q (=> x = Q^-1 q) which has a complexity of O(n^3) otherwise
        """
        x = np.array(x)
        return (0.5 * x.T.dot(self.Q).dot(x) + self.q.T.dot(x)).item() \
            if x.size != 0 else self.function(np.linalg.inv(self.Q).dot(-self.q)) \
            if min(np.linalg.eig(self.Q)[0]) > 1e-14 else -np.inf  # or np.linalg.solve(Q, -q)

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of a general quadratic function J f(x) = Q x + q
        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (quadratic part) of f
        :param q: ([n x 1] real column vector): the linear part of f
        :param x: ([n x 1] real column vector): the point where to start the algorithm from
        :return: the Jacobian of a general quadratic function
        """
        return self.Q.dot(x) + self.q  # complexity O(n^2)

    def hessian(self, x=None):
        """
        The Hessian matrix of a general quadratic function H f(x) = Q
        :param x: 1-D array of points at which the Hessian is to be computed
        :return: the Hessian matrix (i.e. the quadratic part) of a general quadratic function at x
        """
        return self.Q

    def plot(self):
        xmin, xmax, xstep = -100, 100, 100
        ymin, ymax, ystep = -100, 100, 100

        x, y = np.meshgrid(np.arange(xmin, xmax, xstep), np.arange(ymin, ymax, ystep))

        # 3D surface plot
        fig = plt.figure()
        ax = Axes3D(fig)

        # generic quadratic function
        z = 0.5 * (x.T + y.T).dot(self.Q).dot(x.T + y.T) + self.q.T.dot(x.T + y.T)

        ax.plot_surface(x, y, z, norm=LogNorm(), cmap=cm.get_cmap('jet'))

        # 2D contour
        fig, ax = plt.subplots()

        ax.contour(x, y, z, cmap=cm.get_cmap('jet'))
        ax.plot(*np.array([0., 0.]), 'r*', markersize=10)


if __name__ == "__main__":
    Q = [[6, -2], [-2, 6]]
    q = [[10], [10]]

    GenericQuadratic(Q, q).plot()


class Rosenbrock:

    def __init__(self):
        self.rosenbrock_jacobian = jacobian(self.function)
        self.rosenbrock_hessian = hessian(self.function)

    def function(self, x):
        """
        The Rosenbrock function.
        :param x: 1-D array of points at which the Rosenbrock function is to be computed
        :return: the value of the Rosenbrock function at x
        """
        x = np.array(x)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of the Rosenbrock function.
        :param x: 1-D array of points at which the Jacobian is to be computed
        :return: the Jacobian of the Rosenbrock function at x
        """
        return self.rosenbrock_jacobian(np.array(x, dtype=float))

    def hessian(self, x):
        """
        The Hessian matrix of the Rosenbrock function.
        :param x: 1-D array of points at which the Hessian is to be computed
        :return: the Hessian matrix of the Rosenbrock function at x
        """
        return self.rosenbrock_jacobian(np.array(x, dtype=float))

    def plot(self):
        xmin, xmax, xstep = -2, 2, 0.1
        ymin, ymax, ystep = -1, 3, 0.1

        x, y = np.meshgrid(np.arange(xmin, xmax, xstep), np.arange(ymin, ymax, ystep))

        # 3D surface plot
        surface_plot = plt.figure()
        surface_axes = Axes3D(surface_plot)

        # Rosenbrock function
        z = 100. * (y - x ** 2) ** 2 + (1. - x) ** 2

        surface_axes.plot_surface(x, y, z, norm=LogNorm(), cmap=cm.get_cmap('jet'))

        # 2D contour
        contour_plot, contour_axes = plt.subplots()

        contour_axes.contour(x, y, z, cmap=cm.get_cmap('jet'))
        contour_axes.plot(*np.array([1., 1.]), 'r*', markersize=10)

        return surface_plot, surface_axes, contour_plot, contour_axes


class Ackley:

    def __init__(self):
        self.ackley_jacobian = jacobian(self.function)
        self.ackley_hessian = hessian(self.function)

    def function(self, x):
        """
        The Ackley function.
        :param x: 1-D array of points at which the Ackley function is to be computed
        :return: the value of the Ackley function
        """
        x = np.array(x)
        sum_sq_term = -0.2 * np.sqrt(np.sum(x ** 2) / len(x))
        sum_cos_term = np.sum(np.cos(2 * np.pi * x)) / len(x)
        return -20 * np.exp(sum_sq_term) - np.exp(sum_cos_term) + 20 + np.e

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of the Ackley function.
        :param x: 1-D array of points at which the Jacobian is to be computed
        :return: the Jacobian of the Ackley function at x
        """
        return self.ackley_jacobian(np.array(x, dtype=float))

    def hessian(self, x):
        """
        The Hessian matrix of the Ackley function.
        :param x: 1-D array of points at which the Hessian is to be computed
        :return: the Hessian matrix of the Ackley function at x
        """
        return self.ackley_hessian(np.array(x, dtype=float))

    def plot(self):
        xmin, xmax, xstep = -32, 32, 0.1
        ymin, ymax, ystep = -32, 32, 0.1

        x, y = np.meshgrid(np.arange(xmin, xmax, xstep), np.arange(ymin, ymax, ystep))

        # 3D surface plot
        surface_plot = plt.figure()
        surface_axes = Axes3D(surface_plot)

        # Ackley function
        sum_sq_term = -0.2 * np.sqrt(x ** 2 + y ** 2) / 2
        cos_term = (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x)) / 2
        z = -20 * np.exp(sum_sq_term) - np.exp(cos_term) + 20 + np.e

        surface_axes.plot_surface(x, y, z, norm=LogNorm(), cmap=cm.get_cmap('jet'))

        # 2D contour
        contour_plot, contour_axes = plt.subplots()

        contour_axes.contour(x, y, z, cmap=cm.get_cmap('jet'))
        contour_axes.plot(*np.array([0., 0.]), 'r*', markersize=10)

        return surface_plot, surface_axes, contour_plot, contour_axes
