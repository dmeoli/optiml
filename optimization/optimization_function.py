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
        Construct a quadratic function from its linear and quadratic part defined as:

                                    1/2 x^T Q x - q^T x

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
        return self.function(self.x_star())

    def function(self, x):
        """
        A general quadratic function f(x) = 1/2 x^T Q x - q^T x.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the value of a general quadratic function if x, the optimal solution of a
                  linear system Qx = q (=> x = Q^-1 q) which has a complexity of O(n^3) otherwise.
        """
        return 0.5 * x.T.dot(self.Q).dot(x) - self.q.T.dot(x)

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of a general quadratic function J f(x) = Q x - q.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the Jacobian of a general quadratic function.
        """
        return self.Q.dot(x) - self.q

    def hessian(self, x):
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
        z = (0.5 * self.Q[0][0] * x ** 2 + self.Q[0][1] * x * y +
             0.5 * self.Q[1][1] * y ** 2 - self.q[0] * x - self.q[1] * y)

        surface_axes.plot_surface(x, y, z, norm=LogNorm(), cmap=cm.get_cmap('jet'))

        # 2D contour
        contour_plot, contour_axes = plt.subplots()

        contour_axes.contour(x, y, z, cmap=cm.get_cmap('jet'))
        contour_axes.plot(*self.x_star(), 'r*', markersize=10)
        return surface_plot, surface_axes, contour_plot, contour_axes


# 2x2 quadratic function with nicely conditioned Hessian
quad1 = Quadratic([[6, -2], [-2, 6]], [10, 5])
# 2x2 quadratic function with less nicely conditioned Hessian
quad2 = Quadratic([[5, -3], [-3, 5]], [10, 5])
# 2x2 quadratic function with Hessian having one zero eigenvalue (singular matrix)
quad3 = Quadratic([[4, -4], [-4, 4]], [10, 5])
# 2x2 quadratic function with indefinite Hessian (one positive and one negative eigenvalue)
quad4 = Quadratic([[3, -5], [-5, 3]], [10, 5])
# 2x2 quadratic function with "very elongated" Hessian
# (a very small positive minimum eigenvalue, the other much larger)
quad5 = Quadratic([[101, -99], [-99, 101]], [10, 5])


class BoxConstrainedQuadratic(Quadratic):

    def __init__(self, Q=None, q=None, ub=None, n=10, actv=0.5, rank=1.1, ecc=0.99, u_min=8, u_max=12):
        """
        Construct a box-constrained quadratic function defined as:

                    1/2 x^T Q x + q^T x : 0 <= x <= ub

        :param Q:  ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                            the Hessian (i.e., the quadratic part) of f. If it is not
                            positive semidefinite, f(x) will be unbounded below.
        :param q:  ([n x 1] real column vector): the linear part of f.
        :param ub: ([n x 1] real column vector): the upper bound of the box.
        """
        if Q is None and q is None:
            if not np.isscalar(n) or not np.isreal(n):
                raise ValueError('n not a real scalar')
            n = round(n)
            if n <= 0:
                raise ValueError('n must be > 0')
            if not np.isscalar(actv) or not np.isreal(actv):
                raise ValueError('actv not a real scalar')
            if not 0 <= actv <= 1:
                raise ValueError('actv must be in [0, 1]')
            if not np.isscalar(rank) or not np.isreal(rank):
                raise ValueError('rank not a real scalar')
            if rank <= 0:
                raise ValueError('rank must be > 0')
            if not np.isscalar(ecc) or not np.isreal(ecc):
                raise ValueError('ecc not a real scalar')
            if not 0 <= ecc < 1:
                raise ValueError('ecc must be in [0, 1)')
            if not np.isscalar(u_min) or not np.isreal(u_min):
                raise ValueError('u_min not a real scalar')
            if u_min <= 0:
                raise ValueError('u_min must be > 0')
            if not np.isscalar(u_max) or not np.isreal(u_max):
                raise ValueError('u_min not a real scalar')
            if u_max <= u_min:
                raise ValueError('u_max must be > u_min')

            ub = u_min * np.ones(n) + (u_max - u_min) * np.random.rand(n)

            G = np.random.rand(round(rank * n), n)
            Q = G.T.dot(G)

            # compute eigenvalue decomposition
            D, V = np.linalg.eigh(Q)  # V.dot(np.diag(D)).dot(V.T) = Q

            if D[0] > 1e-14:  # smallest eigenvalue
                # modify eccentricity only if \lambda_n > 0, for when \lambda_n = 0 the
                # eccentricity is 1 by default. The formula is:
                #
                #                         \lambda_i - \lambda_n           2 ecc
                # \lambda_i = \lambda_n + --------------------- \lambda_n -------
                #                         \lambda_1 - \lambda_n           1 - ecc
                #
                # This leaves \lambda_n unchanged, and modifies all the other ones
                # proportionally so that:
                #
                #   \lambda_1 - \lambda_n
                #   --------------------- = ecc
                #   \lambda_1 - \lambda_n

                l = D[0] + (D[0] / (D[-1] - D[0])) * (2 * ecc / (1 - ecc)) * (D - D[0])

                Q = V.dot(np.diag(l)).dot(V.T)

            # we first generate the unconstrained minimum z of the problem in the form:
            #
            #          min 1/2 (x - z)^T Q (x - z)
            #    min 1/2 x^T Q x - z^T Q x + 1/2 z^T Q z
            #
            # and then we set q = -z^T Q

            z = np.zeros(n)

            # out_b[i] = True if z[i] will be out of the bounds
            out_b = np.random.rand(n) <= actv

            # 50/50 chance of being left of lb or right of ub
            lr = np.random.rand(n) <= 0.5
            l = np.logical_and(out_b, lr)
            r = np.logical_and(out_b, np.logical_not(lr))

            # a random amount left of the lb[0]
            z[l] = -np.random.rand(sum(l)) * ub[l]

            # a random amount right of the ub[u]
            z[r] = ub[r] * (1 + np.random.rand(sum(r)))

            out_b = np.logical_not(out_b)  # entries that will be inside the bound
            # pick at random in [0, u]
            z[out_b] = np.random.rand(sum(out_b)) * ub[out_b]

            q = -Q.dot(z)

        super().__init__(Q, -q)
        self.ub = ub

    def plot(self, x_min=-5, x_max=2, y_min=-5, y_max=2):
        # TODO add box-constraints
        pass


class LagrangianBoxConstrained(Quadratic):

    def __init__(self, f):
        """
        Construct the lagrangian relaxation of a box-constrained quadratic function defined as:

                        1/2 x^T Q x + q^T x - lambda^+ (u - x) - lambda^- x
                    1/2 x^T Q x + (q^T + lambda^+ - lambda^-) x - lambda^+ u

        where lambda^+ are the first n components of lmbda, and lambda^- the last n components;
        both are constrained to be >= 0.
        :param f: box-constrained quadratic function to be relaxed
        """
        if not isinstance(f, BoxConstrainedQuadratic):
            raise TypeError('f is not a box-constrained quadratic function')
        super().__init__(f.Q, f.q)
        self.primal = f
        # compute the LDL^T Cholesky indefinite factorization of Q, this will
        # be used at each iteration to solve the Lagrangian relaxation
        # TODO solve the system with LDL^T Cholesky indefinite factorization or with null space method
        # self.R = ldl(self.f.Q)
        try:
            self.R = np.linalg.cholesky(self.Q)
        except np.linalg.LinAlgError:
            raise ValueError('Q is not positive definite, this is not yet supported')

    def function(self, lmbda):
        """
        The optimal solution of the Lagrangian relaxation is the (unique)
        solution of the linear system:

                        Q x = -q - lambda^+ + lambda^-

        Since we have saved the LDL^T Cholesky factorization of Q,
        i.e., Q = R^T R, where R is lower triangular and therefore R^T is upper
        triangular, we obtain this by just two triangular backsolves:

                        R z = -q - lambda^+ + lambda^-

                                 R^T x = z

        return the function value and the primal solution, respectively.
        :param lmbda:
        :return: the function value
        """

        ql = -self.q + lmbda[:self.n] - lmbda[self.n:]
        # TODO solve the systems with LDL^T Cholesky indefinite factorization or with null space method
        z = np.linalg.solve(self.R, -ql)
        y = np.linalg.solve(self.R.T, z)
        # z = solve_triangular(self.R.T, -ql, lower=True)
        # y = solve_triangular(self.R, z, lower=False)

        return -((0.5 * y.T.dot(self.Q) + ql.T).dot(y) - lmbda[:self.n].T.dot(self.primal.ub))

    def jacobian(self, lmbda):
        """
        Compute the jacobian of the lagrangian relaxation as follow: with x the optimal
        solution of the minimization problem, the gradient at lmbda is [x - u, -x].
        However, we rather want to maximize the lagrangian relaxation, hence we have to
        change the sign of both function values and gradient entries.
        :param lmbda:
        :return:
        """

        ql = -self.q + lmbda[:self.n] - lmbda[self.n:]
        # TODO solve the systems with LDL^T Cholesky indefinite factorization or with null space method
        z = np.linalg.solve(self.R, -ql)
        y = np.linalg.solve(self.R.T, z)
        # z = solve_triangular(self.R.T, -ql, lower=True)
        # y = solve_triangular(self.R, z, lower=False)

        return np.hstack((self.primal.ub - y, y))


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
