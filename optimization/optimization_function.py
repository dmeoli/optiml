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
        self.n = n

    def x_star(self):
        raise NotImplementedError

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
        return np.linalg.inv(self.Q).dot(self.q)  # or np.linalg.solve(self.Q, self.q)

    def f_star(self):
        return self.function(self.x_star(), *self.args())

    def args(self):
        return self.Q, self.q

    def function(self, x, Q=None, q=None):
        """
        A general quadratic function f(x) = 1/2 x^T Q x - q^T x.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (i.e. the quadratic part) of f. If it is not
                           positive semidefinite, f(x) will be unbounded below.
        :param q: ([n x 1] real column vector): the linear part of f.
        :return:  the value of a general quadratic function if x, the optimal solution of a
                  linear system Qx = q (=> x = Q^-1 q) which has a complexity of O(n^3) otherwise.
        """
        Q, q = Q if Q is not None else self.Q, q if q is not None else self.q
        return 0.5 * x.T.dot(Q).dot(x) - q.T.dot(x)

    def jacobian(self, x, Q=None, q=None):
        """
        The Jacobian (i.e. gradient) of a general quadratic function J f(x) = Q x - q.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (i.e. the quadratic part) of f. If it is not
                           positive semidefinite, f(x) will be unbounded below.
        :param q: ([n x 1] real column vector): the linear part of f.
        :return:  the Jacobian of a general quadratic function.
        """
        Q, q = Q if Q is not None else self.Q, q if q is not None else self.q
        return Q.dot(x) - q

    def hessian(self, x, Q=None, q=None):
        """
        The Hessian matrix of a general quadratic function H f(x) = Q.
        :param x: 1D array of points at which the Hessian is to be computed.
        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (i.e. the quadratic part) of f. If it is not
                           positive semidefinite, f(x) will be unbounded below.
        :param q: ([n x 1] real column vector): the linear part of f.
        :return:  the Hessian matrix (i.e. the quadratic part) of a general quadratic function at x.
        """
        return Q if Q is not None else self.Q

    def plot(self, x_min=-5, x_max=2, y_min=-5, y_max=2):
        x, y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # 3D surface plot
        surface_plot = plt.figure()
        surface_axes = Axes3D(surface_plot)

        #                      T                           T
        # f(x, y) = 1/2 * | x |  * | a  b | * | x | - | d |  * | x |
        #                 | y |    | b  c |   | y |   | e |    | y |
        z = (0.5 * self.Q[0][0] * x ** 2 + self.Q[0][1] * x * y +
             0.5 * self.Q[1][1] * y ** 2 - self.q[0] * x - self.q[1] * y)

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


class BoxConstrainedQuadratic(Quadratic):
    # Produces a structure encoding a convex Box-Constrained Quadratic program:
    #
    #  (P) min { 1/2 x^T Q x + q^T x : 0 <= x <= ub }

    def __init__(self, Q, q, ub):
        super().__init__(Q, q)
        self.ub = ub

    def plot(self, x_min=-5, x_max=2, y_min=-5, y_max=2):
        # TODO add box-constraints
        pass


class Rosenbrock(OptimizationFunction):

    def __init__(self, n=2, a=1, b=2):
        super().__init__(n)
        self.a = a
        self.b = b

    def x_star(self):
        return np.zeros(self.n) if self.a is 0 else np.ones(self.n)

    def f_star(self):
        return self.function(self.x_star())

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


class Ackley(OptimizationFunction):

    def __init__(self, n=2):
        super().__init__(n)

    def x_star(self):
        return np.zeros(self.n)

    def f_star(self):
        return self.function(self.x_star())

    def function(self, x):
        """
        The Ackley function.
        :param x: 1D array of points at which the Ackley function is to be computed.
        :return:  the value of the Ackley function.
        """
        x = np.array(x)
        return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / x.size)) -
                np.exp((np.sum(np.cos(2.0 * np.pi * x))) / x.size) + np.e + 20)

    def plot(self, x_min=-5, x_max=5, y_min=-5, y_max=5):
        x, y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # 3D surface plot
        surface_plot = plt.figure()
        surface_axes = Axes3D(surface_plot)

        # Ackley function
        z = (-20 * np.exp(-0.2 * np.sqrt((x ** 2 + y ** 2) * 0.5)) -
             np.exp((np.cos(2.0 * np.pi * x) + np.cos(2 * np.pi * y)) * 0.5) + np.e + 20)

        surface_axes.plot_surface(x, y, z, norm=LogNorm(), cmap=cm.get_cmap('jet'))

        # 2D contour
        contour_plot, contour_axes = plt.subplots()

        contour_axes.contour(x, y, z, cmap=cm.get_cmap('jet'))
        contour_axes.plot(*self.x_star(), 'r*', markersize=10)

        return surface_plot, surface_axes, contour_plot, contour_axes
