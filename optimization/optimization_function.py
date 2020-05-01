import autograd.numpy as np
from autograd import jacobian, hessian
from matplotlib import pyplot as plt
from scipy.linalg import ldl

from utils import ldl_solve


class OptimizationFunction:

    def __init__(self, ndim, x_min, x_max, y_min, y_max):
        self.jac = jacobian(self.function)
        self.hes = hessian(self.function)
        self.n = ndim
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

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

    def plot(self):
        X, Y = np.meshgrid(np.arange(self.x_min, self.x_max, 0.1), np.arange(self.y_min, self.y_max, 0.1))

        Z = np.array([self.function(np.array([x, y]))
                      for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

        fig = plt.figure(figsize=(16, 8))

        # 3D surface plot
        ax = fig.add_subplot(1, 2, 1, projection='3d', elev=50, azim=-50)
        ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.5)
        ax.plot([self.x_star()[0]], [self.x_star()[1]], [self.f_star()], marker='*', color='r', markersize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x)$')

        # 2D contour plot
        ax = fig.add_subplot(1, 2, 2)
        ax.contour(X, Y, Z, 70, cmap='jet')
        ax.plot(*self.x_star(), marker='*', color='r', markersize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        return fig


class Quadratic(OptimizationFunction):

    def __init__(self, Q, q, x_min=-5, x_max=2, y_min=-5, y_max=2):
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
        super().__init__(n, x_min, x_max, y_min, y_max)

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
            # alternatively we can solve the linear system as np.linalg.inv(self.Q).dot(-self.q)
            # but the complexity increase about 3 times as much as LU factorization used by
            # default in np.linalg.solve(self.Q, -self.q) because it requires is O(2n^3) to
            # compute the inverse of the Hessian matrix and O(2n^2) to multiply this by the -q vector
            return np.linalg.solve(self.Q, -self.q)  # complexity O(2n^3/3)
        except np.linalg.LinAlgError:  # the Hessian matrix is singular
            return np.full(self.n, np.inf)

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


class BoxConstrained(Quadratic):

    def __init__(self, Q=None, q=None, ub=None, x_min=-5, x_max=2, y_min=-5, y_max=2,
                 ndim=2, actv=0.5, rank=1.1, ecc=0.99, u_min=8, u_max=12):
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
            if not np.isscalar(ndim) or not np.isreal(ndim):
                raise ValueError('n not a real scalar')
            ndim = round(ndim)
            if ndim <= 0:
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

            np.random.seed(ndim)

            ub = u_min * np.ones(ndim) + (u_max - u_min) * np.random.rand(ndim)

            G = np.random.rand(round(rank * ndim), ndim)
            Q = G.T.dot(G)

            # compute eigenvalue decomposition
            D, V = np.linalg.eigh(Q)  # V.dot(np.diag(D)).dot(V.T) = Q

            if min(D) > 1e-14:  # smallest eigenvalue
                # modify eccentricity only if \lambda_n > 0, for when \lambda_n = 0 the
                # eccentricity is 1 by default. The formula is:
                #
                #                         \lambda_i - \lambda_n            2 ecc
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

            z = np.zeros(ndim)

            # out_b[i] = True if z[i] will be out of the bounds
            out_b = np.random.rand(ndim) <= actv

            # 50/50 chance of being left of lb or right of ub
            lr = np.random.rand(ndim) <= 0.5
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

        super().__init__(Q, q, x_min, x_max, y_min, y_max)
        self.ub = ub


class LagrangianBoxConstrained(Quadratic):

    def __init__(self, f, x_min=-5, x_max=2, y_min=-5, y_max=2):
        """
        Construct the lagrangian relaxation of a box-constrained quadratic function defined as:

                       1/2 x^T Q x + q^T x - lambda^+ (u - x) - lambda^- x
                    1/2 x^T Q x + (q^T + lambda^+ - lambda^-) x - lambda^+ u

        where lambda^+ are the first n components of lmbda, and lambda^- the last n components;
        both are constrained to be >= 0.
        :param f: box-constrained quadratic function to be relaxed
        """
        if not isinstance(f, BoxConstrained):
            raise TypeError('f is not a box-constrained quadratic function')
        super().__init__(f.Q, f.q, x_min, x_max, y_min, y_max)
        self.n *= 2
        # Compute the LDL^T Cholesky symmetric indefinite factorization
        # of Q because it is symmetric but could be not positive definite.
        # This will be used at each iteration to solve the Lagrangian relaxation.
        self.L, self.D, self.P = ldl(self.Q)
        self.primal = f
        self.primal_solution = np.inf
        self.primal_value = np.inf

    def function(self, lmbda):
        """
        The optimal solution of the Lagrangian relaxation is the unique
        solution of the linear system:

                        Q x = -q - lambda^+ + lambda^-

        Since we have saved the LDL^T Cholesky factorization of Q,
        i.e., Q = L D L^T, we obtain this by solving:

                     L D L^T x = -q - lambda^+ + lambda^-

        :param lmbda:
        :return: the function value
        """
        ql = self.q.T + lmbda[:self.primal.n] - lmbda[self.primal.n:]
        x = ldl_solve((self.L, self.D, self.P), -ql)
        return (0.5 * x.T.dot(self.Q) + ql.T).dot(x) - lmbda[:self.primal.n].dot(self.primal.ub)

    def jacobian(self, lmbda):
        """
        Compute the jacobian of the lagrangian relaxation as follow: with x the optimal
        solution of the minimization problem, the gradient at lmbda is [x - u, -x].
        However, we rather want to maximize the lagrangian relaxation, hence we have to
        change the sign of both function values and gradient entries.
        :param lmbda:
        :return:
        """
        ql = self.q.T + lmbda[:self.primal.n] - lmbda[self.primal.n:]
        x = ldl_solve((self.L, self.D, self.P), -ql)
        g = np.hstack((self.primal.ub - x, x))

        # compute an heuristic solution out of the solution x of
        # the Lagrangian relaxation by projecting x on the box
        x[x < 0] = 0
        idx = x > self.primal.ub
        x[idx] = self.primal.ub[idx]

        v = self.primal.function(x)
        if v < self.primal_value:
            self.primal_solution = x
            self.primal_value = v

        return g


class Rosenbrock(OptimizationFunction):

    def __init__(self, ndim=2, a=1, b=2, x_min=-2, x_max=2, y_min=-1, y_max=3):
        super().__init__(ndim, x_min, x_max, y_min, y_max)
        self.a = a
        self.b = b

    def x_star(self):
        return np.zeros(self.n) if self.a == 0 else np.ones(self.n)

    def f_star(self):
        return self.function(self.x_star())

    def function(self, x):
        """
        The Rosenbrock function.
        :param x: 1D array of points at which the Rosenbrock function is to be computed.
        :return:  the value of the Rosenbrock function at x.
        """
        return np.sum(self.b * (x[1:] - x[:-1] ** 2) ** 2 + (self.a - x[:-1]) ** 2)
