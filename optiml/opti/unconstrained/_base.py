import autograd.numpy as np

from .. import OptimizationFunction


class Rosenbrock(OptimizationFunction):

    def __init__(self, ndim=2, a=1, b=2):
        super(Rosenbrock, self).__init__(ndim)
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


class Ackley(OptimizationFunction):

    def x_star(self):
        return np.zeros(2)

    def f_star(self):
        return self.function(self.x_star())

    def function(self, x):
        """
        The Ackley function.
        :param x: 1D array of points at which the Ackley function is to be computed.
        :return:  the value of the Ackley function.
        """
        return (-20 * np.exp(-0.2 * np.sqrt(np.sum(np.square(x)) / 2)) -
                np.exp((np.sum(np.cos(2 * np.pi * x))) / 2) + np.e + 20)


class SixHumpCamel(OptimizationFunction):

    def x_star(self):
        return np.array([[-0.0898, 0.0898],
                         [0.7126, -0.7126]])

    def f_star(self):
        return (self.function(self.x_star()[:, 0]) or
                self.function(self.x_star()[:, 1]))

    def function(self, x):
        """
        The Six-Hump Camel function.
        :param x: 1D array of points at which the Six-Hump Camel function is to be computed.
        :return:  the value of the Six-Hump Camel function.
        """
        return ((4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2 +
                x[0] * x[1] + (-4 + 4 * x[1] ** 2) * x[1] ** 2)
