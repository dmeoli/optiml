import numpy as np
from scipy.linalg import ldl

from .. import Optimizer
from ..unconstrained import Quadratic
from ..utils import ldl_solve


class BoxConstrainedQuadraticOptimizer(Optimizer):
    def __init__(self, f, eps=1e-6, max_iter=1000, callback=None, callback_args=(), verbose=False):
        if not isinstance(f, BoxConstrainedQuadratic):
            raise TypeError(f'{f} is not an allowed box-constrained quadratic function')
        super().__init__(f, f.ub / 2,  # starts from the middle of the box
                         eps, max_iter, callback, callback_args, verbose)

    def minimize(self):
        raise NotImplementedError


class BoxConstrainedQuadratic(Quadratic):

    def __init__(self, Q=None, q=None, ub=None, ndim=2, actv=0.5, rank=1.1, ecc=0.99, ub_min=8, ub_max=12, seed=None):
        """
        Construct a box-constrained quadratic function defined as:

                    1/2 x^T Q x + q^T x : 0 <= x <= ub

        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (i.e., the quadratic part) of f. If it is not
                           positive semidefinite, f(x) will be unbounded below.
        :param q: ([n x 1] real column vector): the linear part of f.
        :param ub: ([n x 1] real column vector): the upper bound of the box.
        :param ndim: (integer, scalar): the size of the problem
        :param actv: (real, scalar, default 0.5): how many box constraints (as a
                     fraction of the number of variables n of the problems) the
                     unconstrained optimum will violate, and therefore we expect to be
                     active in the constrained optimum; note that there is no guarantee that
                     exactly acvt constraints will be active, they may be less or (more
                     likely) more, except when actv = 0 because then the unconstrained
                     optimum is surely feasible and therefore it will be the constrained
                     optimum as well
        :param rank: (real, scalar, default 1.1): Q will be obtained as Q = G^T G, with
                     G a m \times n random matrix with m = rank * n. If rank > 1 then Q can
                     be expected to be full-rank, if rank < 1 it will not
        :param ecc: (real, scalar, default 0.99): the eccentricity of Q, i.e., the
                    ratio ( \lambda_1 - \lambda_n ) / ( \lambda_1 + \lambda_n ), with
                    \lambda_1 the largest eigenvalue and \lambda_n the smallest one. Note
                    that this makes sense only if \lambda_n > 0, for otherwise the
                    eccentricity is always 1; hence, this setting is ignored if
                    \lambda_n = 0, i.e., Q is not full-rank (see above). An eccentricity of
                    0 means that all eigenvalues are equal, as eccentricity -> 1 the
                    largest eigenvalue gets larger and larger w.r.t. the smallest one
        :param seed: (integer, default 0): the seed for the random number generator
        :param ub_min: (real, scalar, default 8): the minimum value of each ub_i
        :param ub_max: (real, scalar, default 12): the maximum value of each ub_i
        """
        if Q is None and q is None:
            if not ndim >= 2:
                raise ValueError('ndim must be >= 2')
            ndim = round(ndim)
            if ndim <= 0:
                raise ValueError('n must be > 0')
            if not 0 <= actv <= 1:
                raise ValueError('actv has to lie in [0, 1]')
            if not rank > 0:
                raise ValueError('rank must be > 0')
            if not 0 <= ecc < 1:
                raise ValueError('ecc has to lie in [0, 1)')
            if not ub_min > 0:
                raise ValueError('ub_min must be > 0')
            if ub_max <= ub_min:
                raise ValueError('ub_max must be > ub_min')

            np.random.seed(seed)

            ub = ub_min * np.ones(ndim) + (ub_max - ub_min) * np.random.rand(ndim)

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
        else:
            if any(u < 0 for u in ub):
                raise ValueError('the lower bound must be > 0')

        super().__init__(Q, q)
        self.ub = np.asarray(ub, dtype=np.float)


class LagrangianBoxConstrainedQuadratic(BoxConstrainedQuadratic):

    def __init__(self, bcq):
        """
        Construct the lagrangian relaxation of a constrained quadratic function defined as:
                           
                           1/2 x^T Q x + q^T x : 0 <= x <= ub
                           
                       1/2 x^T Q x + q^T x - lambda^+ (ub - x) - lambda^- x
                    1/2 x^T Q x + (q^T + lambda^+ - lambda^-) x - lambda^+ ub

        where lambda^+ are the first n components of lambda, and lambda^- the last n components;
        both are constrained to be >= 0.
        :param bcq: constrained quadratic function to be relaxed
        """
        if not isinstance(bcq, BoxConstrainedQuadratic):
            raise TypeError(f'{bcq} is not an allowed quadratic function')
        super().__init__(bcq.Q, bcq.q, bcq.ub)
        self.ndim *= 2
        # Compute the LDL^T Cholesky symmetric indefinite factorization
        # of Q because it is symmetric but could be not positive definite.
        # This will be used at each iteration to solve the Lagrangian relaxation.
        self.L, self.D, self.P = ldl(self.Q)
        self.primal = bcq
        self.primal_solution = np.inf
        self.primal_value = np.inf

    def x_star(self):
        raise np.full(fill_value=np.nan, shape=self.ndim)

    def f_star(self):
        return np.inf

    def function(self, lmbda):
        """
        The optimal solution of the Lagrangian relaxation is the unique
        solution of the linear system:

                        Q x = q^T + lambda^+ - lambda^-

        Since we have saved the LDL^T Cholesky factorization of Q,
        i.e., Q = L D L^T, we obtain this by solving:

                     L D L^T x = q^T + lambda^+ - lambda^-

        :param lmbda:
        :return: the function value
        """
        ql = self.q.T + lmbda[:self.primal.ndim] - lmbda[self.primal.ndim:]
        x = ldl_solve((self.L, self.D, self.P), -ql)
        return (0.5 * x.T.dot(self.Q) + ql.T).dot(x) - lmbda[:self.primal.ndim].T.dot(self.primal.ub)

    def jacobian(self, lmbda):
        """
        Compute the jacobian of the lagrangian relaxation as follow: with x the optimal
        solution of the minimization problem, the gradient at lambda is [x - u, -x].
        However, we rather want to maximize the lagrangian relaxation, hence we have to
        change the sign of both function values and gradient entries.
        :param lmbda:
        :return:
        """
        ql = self.q.T + lmbda[:self.primal.ndim] - lmbda[self.primal.ndim:]
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
