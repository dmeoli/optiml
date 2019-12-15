import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generic_quad(Q, q, x):
    """
    A general quadratic function f(x) = 1/2 x^T Q x + q x.
    :param Q: ([ n x n ] real symmetric matrix, not necessarily positive semidefinite):
                         the Hessian (quadratic part) of f. If it is not positive
                         semidefinite, f(x) will be unbounded below
    :param q: ([ n x 1 ] real column vector): the linear part of f
    :param x: ([ n x 1 ] real column vector): the point where to start the algorithm from
    :return: the value of a general quadratic function
    """
    Q = np.asarray(Q)
    q = np.asarray(q)
    x = np.asarray(x)
    return 0.5 * x.T.dot(Q).dot(x) + q.T.dot(x)


def generic_quad_der(Q, q, x):
    """
    The derivative (i.e. gradient) of a general quadratic function.
    :param Q: ([ n x n ] real symmetric matrix, not necessarily positive semidefinite):
                         the Hessian (quadratic part) of f
    :param q: ([ n x 1 ] real column vector): the linear part of f
    :param x: ([ n x 1 ] real column vector): the point where to start the algorithm from
    :return: the gradient of a general quadratic function
    """
    Q = np.asarray(Q)
    q = np.asarray(q)
    x = np.asarray(x)
    return Q.dot(x) + q  # complexity O(n^2)


class Rosenbrock:

    def function(self, x):
        """
        The Rosenbrock function.
        :param x: 1-D array of points at which the Rosenbrock function is to be computed
        :return: the value of the Rosenbrock function at x
        """
        x = np.asarray(x)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) if x.size != 0 else 0

    def derivative(self, x):
        """
        The derivative (i.e. gradient) of the Rosenbrock function.
        :param x: 1-D array of points at which the derivative is to be computed
        :return: the gradient of the Rosenbrock function at x
        """
        x = np.asarray(x)
        if x.size != 0:
            xm = x[1:-1]
            xm_m1 = x[:-2]
            xm_p1 = x[2:]
            der = np.zeros_like(x)
            der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
            der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
            der[-1] = 200 * (x[-1] - x[-2] ** 2)
            return der
        return np.asarray([[-1], [1]])

    def hessian(self, x):
        """
        The Hessian matrix of the Rosenbrock function.
        :param x: 1-D array of points at which the derivative is to be computed
        :return: the Hessian matrix of the Rosenbrock function at x
        """
        x = np.asarray(x)
        H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
        diagonal = np.zeros_like(x)
        diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
        H = H + np.diag(diagonal)
        return H

    def plot(self):
        fig = plt.figure()

        ax = Axes3D(fig)
        x = np.arange(-32, 32, 0.25)
        y = np.arange(-32, 32, 0.25)
        x, y = np.meshgrid(x, y)
        z = 100. * (y - x ** 2) ** 2 + (1. - x) ** 2

        # sum_sq_term = np.exp(-0.2 * np.sqrt(X * X + Y * Y) / 2)
        # cos_term = np.exp((np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)) / 2)
        # Z = -20 * sum_sq_term - cos_term + 20 + np.e

        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


def ackley(x):
    """
    The Ackley function.
    :param x: 1-D array of points at which the Rosenbrock function is to be computed
    :return: the value of the Rosenbrock function
    """
    x = np.asarray(x)
    sum_sq_term = -0.2 * np.sqrt(np.sum(x ** 2) / len(x))
    sum_cos_term = np.sum(np.cos(2 * np.pi * x)) / len(x)
    return -20 * np.exp(sum_sq_term) - np.exp(sum_cos_term) + 20 + np.e
