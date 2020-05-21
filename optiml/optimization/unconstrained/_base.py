import autograd.numpy as np

from .. import OptimizationFunction


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

        n = Q.shape[1]
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
        return ((4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.) * x[0] ** 2 +
                x[0] * x[1] + (-4 + 4 * x[1] ** 2) * x[1] ** 2)
