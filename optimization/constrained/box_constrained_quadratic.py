import numpy as np


class BoxConstrainedQuadratic:
    # Produces a structure encoding a convex Box-Constrained Quadratic program
    #
    #  (P) min { (1/2) x^T * Q * x + q * x : 0 <= x <= u }
    #
    # Input:
    #
    # - n (integer, scalar): the size of the problem
    #
    # - actv (real, scalar, default 0.5): how many box constraints (as a
    #   fraction of the number of variables n of the problems) the
    #   unconstrained optimum will violate, and therefore we expect to be
    #   active in the constrained optimum; note that there is no guarantee that
    #   exactly acvt constraints will be active, they may be less or (more
    #   likely) more, except when actv = 0 because then the unconstrained
    #   optimum is surely feasible and therefore it will be the constrained
    #   optimum as well
    #
    # - rank (real, scalar, default 1.1): Q will be obtained as Q = G^T G, with
    #   G a m \times n random matrix with m = rank * n. If rank > 1 then Q can
    #   be expected to be full-rank, if rank < 1 it will not
    #
    # - ecc (real, scalar, default 0.99): the eccentricity of Q, i.e., the
    #   ratio ( \lambda_1 - \lambda_n ) / ( \lambda_1 + \lambda_n ), with
    #   \lambda_1 the largest eigenvalue and \lambda_n the smallest one. Note
    #   that this makes sense only if \lambda_n > 0, for otherwise the
    #   eccentricity is always 1; hence, this setting is ignored if
    #   \lambda_n = 0, i.e., Q is not full-rank (see above). An eccentricity of
    #   0 means that all eigenvalues are equal, as eccentricity -> 1 the
    #   largest eigenvalue gets larger and larger w.r.t. the smallest one
    #
    # - u_min (real, scalar, default 8): the minimum value of each u_i
    #
    # - u_max (real, scalar, default 12): the maximum value of each u_i
    #
    # Output: the BCQP structure, with the following fields:
    #
    # - BCQP.Q: n \times n symmetric positive semidefinite real matrix
    #
    # - BCQP.q: n \times 1 real vector
    #
    # - BCQP.u: n \times 1 real vector > 0

    def __init__(self, n, actv=0.5, rank=1.1, ecc=0.99, u_min=8, u_max=12):
        if not np.isscalar(n) or not np.isreal(n):
            raise ValueError('n not a real scalar')
        if n <= 0:
            raise ValueError('n must be > 0')
        self.n = n

        if not np.isscalar(actv) or not np.isreal(actv):
            raise ValueError('actv not a real scalar')
        if actv < 0 or actv > 1:
            raise ValueError('actv must be in [0, 1]')
        self.actv = actv

        if not np.isscalar(rank) or not np.isreal(rank):
            raise ValueError('rank not a real scalar')
        if rank <= 0:
            raise ValueError('rank must be > 0')
        self.rank = rank

        if not np.isscalar(ecc) or not np.isreal(ecc):
            raise ValueError('ecc not a real scalar')
        if ecc < 0 or ecc >= 1:
            raise ValueError('ecc must be in [0, 1)')
        self.ecc = ecc

        if not np.isscalar(u_min) or not np.isreal(u_min):
            raise ValueError('u_min not a real scalar')
        if u_min <= 0:
            raise ValueError('u_min must be > 0')
        self.u_min = u_min

        if not np.isscalar(u_max) or not np.isreal(u_max):
            raise ValueError('u_min not a real scalar')
        if u_max <= u_min:
            raise ValueError('u_max must be > u_min')
        self.u_max = u_max

    def generate(self):

        # generate u
        self.u = self.u_min * np.ones((self.n, 1)) + (self.u_max - self.u_min) * np.random.rand(self.n, 1)

        # generate Q
        G = np.random.rand(round(self.rank * self.n), self.n)
        Q = G.T.dot(G)

        # compute eigenvalue decomposition
        V, D = np.linalg.eig(Q)  # V * D * V^T = Q
        V.sort()

        if V[0] > 1e-14:  # smallest eigenvalue
            # modify eccentricity only if \lambda_n > 0, for when \lambda_n = 0 the
            # eccentricity is 1 by default
            #
            # the formula is:
            #
            #                         \lambda_i - \lambda_n             2 * ecc
            # \lambda_i = \lambda_n + --------------------- * \lambda_n -------
            #                         \lambda_1 - \lambda_n             1 - ecc
            #
            # This leaves \lambda_n unchanged, and modifies all the other ones
            # proportionally so that
            #
            #   \lambda_1 - \lambda_n
            #   --------------------- = ecc
            #   \lambda_1 - \lambda_n

            l = V[0] * np.ones((self.n, 1)) + (V[0] / (V[self.n - 1] - V[0])) * (
                    2 * self.ecc / (1 - self.ecc)) * (V - V[0])

            Q = D * np.diag(l) * D.T

        self.Q = Q

        # generate q
        # We first generate the unconstrained minimum z of the problem in the form
        #
        #    min_x (1/2) ( x - z )^T * Q * ( x - z ) =
        #          (1/2) x^T * Q * x - z^T * Q * x + (1/2) z^T * Q * z
        #
        # and then we set q = - z^T Q

        z = np.zeros((self.n, 1))

        # out_b[i] = true if z[i] will be out of the bounds
        out_b = np.random.rand(self.n, 1) <= self.actv

        # 50/50 chance of being left of lb or right of ub
        lr = np.random.rand(self.n, 1) <= 0.5
        l = out_b and lr
        r = out_b and not lr

        # a random amount left of the lb (0)
        z[l].lvalue = -np.random.rand(sum(l), 1) * self.u(l)

        # a random amount right of the ub (u)
        z[r].lvalue = self.u(r) * (1 + np.random.rand(sum(r), 1))

        out_b = not out_b  # entries that will be inside the bound
        # pick at random in [ 0 , u ]
        z[out_b].lvalue = np.random.rand(sum(out_b), 1) * self.u(out_b)

        self.q = -self.Q * z

    def function(self, x, Q, q):
        """
        A general quadratic function f(x) = 1/2 x^T Q x - q^T x.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the value of a general quadratic function if x, the optimal solution of a
                  linear system Qx = q (=> x = Q^-1 q) which has a complexity of O(n^3) otherwise.
        """
        return 0.5 * x.T.dot(Q).dot(x) - q.T.dot(x)

    def jacobian(self, x, Q, q):
        """
        The Jacobian (i.e. gradient) of a general quadratic function J f(x) = Q x - q.
        :param x: ([n x 1] real column vector): the point where to start the algorithm from.
        :return:  the Jacobian of a general quadratic function.
        """
        return Q.dot(x) - q

    def hessian(self, x, Q, q):
        """
        The Hessian matrix of a general quadratic function H f(x) = Q.
        :param x: 1D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix (i.e. the quadratic part) of a general quadratic function at x.
        """
        return Q


if __name__ == '__main__':
    BoxConstrainedQuadratic(10).generate()
