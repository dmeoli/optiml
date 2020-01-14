import autograd.numpy as np
from autograd import jacobian, hessian
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


class Function:
    def __init__(self, n=2):
        self._jacobian = jacobian(self.function)
        self._hessian = hessian(self.function)
        self.n = n
        self.x_star = None

    def function(self, x):
        return NotImplementedError

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of the function.
        :param x: 1D array of points at which the Jacobian is to be computed.
        :return:  the Jacobian of the function at x.
        """
        return self._jacobian(np.array(x, dtype=float))

    def hessian(self, x):
        """
        The Hessian matrix of the function.
        :param x: 1D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix of the function at x.
        """
        return self._hessian(np.array(x, dtype=float)).reshape((x.size, x.size))

    def hessian_product(self, x, p):
        """
        Product of the Hessian matrix of the function with a vector.
        :param x: 1D array of points at which the Hessian matrix is to be computed.
        :param p: 1D array, the vector to be multiplied by the Hessian matrix.
        :return:  the Hessian matrix of the function at x multiplied by the vector p.
        """
        return np.dot(self.hessian(x), p)

    def plot(self, x_min, x_max, y_min, y_max):
        return NotImplementedError


class Quadratic(Function):

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

        n = Q.shape[0]
        super().__init__(n)

        if n <= 1:
            raise ValueError('Q is too small')
        if n != Q.shape[1]:
            raise ValueError('Q is not square')
        self.Q = Q

        if not np.isrealobj(q):
            raise ValueError('q not a real vector')
        if q.size != n:
            raise ValueError('q size does not match with Q')
        self.q = q

        try:
            self.x_star = np.linalg.inv(self.Q).dot(self.q)  # np.linalg.solve(self.Q, self.q)
        except np.linalg.LinAlgError:
            self.x_star = np.full((n,), np.nan)

    def function(self, x):
        """
        A general quadratic function f(x) = 1/2 x^T Q x - q^T x.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the value of a general quadratic function if x, the optimal solution of a
                  linear system Qx = q (=> x = Q^-1 q) which has a complexity of O(n^3) otherwise.
        """
        x = np.array(x)
        return 0.5 * x.T.dot(self.Q).dot(x) - self.q.T.dot(x) if x.size != 0 else 0

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of a general quadratic function J f(x) = Q x - q.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the Jacobian of a general quadratic function.
        """
        return self.Q.dot(x) - self.q

    def hessian(self, x=None):
        """
        The Hessian matrix of a general quadratic function H f(x) = Q.
        :param x: 1D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix (i.e. the quadratic part) of a general quadratic function at x.
        """
        return self.Q

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
        contour_axes.plot(*self.x_star, 'r*', markersize=10)
        return surface_plot, surface_axes, contour_plot, contour_axes


# 2x2 quadratic function with nicely conditioned Hessian
quad1 = Quadratic([[6, -2], [-2, 6]], [10, 5])
# 2x2 quadratic function with less nicely conditioned Hessian
quad2 = Quadratic([[5, -3], [-3, 5]], [10, 5])
# 2x2 quadratic function with Hessian having one zero eigenvalue
quad3 = Quadratic([[4, -4], [-4, 4]], [10, 5])
# 2x2 quadratic function with indefinite Hessian
# (one positive and one negative eigenvalue)
quad4 = Quadratic([[3, -5], [-5, 3]], [10, 5])
# 2x2 quadratic function with "very elongated" Hessian
# (a very small positive minimum eigenvalue, the other much larger)
quad5 = Quadratic([[101, -99], [-99, 101]], [10, 5])


class Rosenbrock(Function):

    def __init__(self, n=2, autodiff=True):
        super().__init__(n)
        self.autodiff = autodiff
        self.x_star = np.ones(n)

    def function(self, x):
        """
        The Rosenbrock function.
        :param x: 1D array of points at which the Rosenbrock function is to be computed.
        :return:  the value of the Rosenbrock function at x.
        >>> x = 0.1 * np.arange(10)
        >>> Rosenbrock().function(x)
        76.56
        """
        x = np.array(x)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2, axis=0)

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of the Rosenbrock function.
        :param x: 1D array of points at which the Jacobian is to be computed.
        :return:  the Jacobian of the function at x.
        >>> x = 0.1 * np.arange(9)
        >>> Rosenbrock(autodiff=True).jacobian(x)
        array([ -2. ,  10.6,  15.6,  13.4,   6.4,  -3. , -12.4, -19.4,  62. ])
        >>> Rosenbrock(autodiff=False).jacobian(x)
        array([ -2. ,  10.6,  15.6,  13.4,   6.4,  -3. , -12.4, -19.4,  62. ])
        """
        if not self.autodiff:
            x = np.array(x, dtype=float)
            xm = x[1:-1]
            xm_m1 = x[:-2]
            xm_p1 = x[2:]
            jac = np.zeros_like(x)
            jac[1:-1] = (200 * (xm - xm_m1 ** 2) -
                         400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm))
            jac[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
            jac[-1] = 200 * (x[-1] - x[-2] ** 2)
            return jac
        return super().jacobian(x)

    def hessian(self, x):
        """
        The Hessian matrix of the Rosenbrock function.
        :param x: 1D array of points at which the Hessian matrix is to be computed.
        :return:  the Hessian matrix of the Rosenbrock function at x.
        >>> x = 0.1 * np.arange(4)
        >>> Rosenbrock(autodiff=True).hessian(x)
        array([[-38.,   0.,   0.,   0.],
               [  0., 134., -40.,   0.],
               [  0., -40., 130., -80.],
               [  0.,   0., -80., 200.]])
        >>> Rosenbrock(autodiff=False).hessian(x)
        array([[-38.,   0.,   0.,   0.],
               [  0., 134., -40.,   0.],
               [  0., -40., 130., -80.],
               [  0.,   0., -80., 200.]])
        """
        if not self.autodiff:
            x = np.atleast_1d(x)
            H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
            diag = np.zeros(len(x), dtype=x.dtype)
            diag[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
            diag[-1] = 200
            diag[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
            H = H + np.diag(diag)
            return H
        return super().hessian(x)

    def hessian_product(self, x, p):
        """
        Product of the Hessian matrix of the Rosenbrock function with a vector.
        :param x: 1D array of points at which the Hessian matrix is to be computed.
        :param p: 1D array, the vector to be multiplied by the Hessian matrix.
        :return:  the Hessian matrix of the Rosenbrock function at x multiplied by the vector p.
        >>> x = 0.1 * np.arange(9)
        >>> p = 0.5 * np.arange(9)
        >>> Rosenbrock(autodiff=True).hessian_product(x, p)
        array([  -0.,   27.,  -10.,  -95., -192., -265., -278., -195., -180.])
        >>> Rosenbrock(autodiff=False).hessian_product(x, p)
        array([  -0.,   27.,  -10.,  -95., -192., -265., -278., -195., -180.])
        """
        if not self.autodiff:
            x = np.atleast_1d(x)
            Hp = np.zeros(len(x), dtype=x.dtype)
            Hp[0] = (1200 * x[0] ** 2 - 400 * x[1] + 2) * p[0] - 400 * x[0] * p[1]
            Hp[1:-1] = (-400 * x[:-2] * p[:-2] +
                        (202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]) * p[1:-1] -
                        400 * x[1:-1] * p[2:])
            Hp[-1] = -400 * x[-2] * p[-2] + 200 * p[-1]
            return Hp
        return super().hessian_product(x, p)

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
        contour_axes.plot(*self.x_star, 'r*', markersize=10)

        return surface_plot, surface_axes, contour_plot, contour_axes