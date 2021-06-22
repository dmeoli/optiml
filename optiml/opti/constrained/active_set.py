import numpy as np
from scipy.linalg import cho_solve, cho_factor
from scipy.sparse.linalg import minres

from optiml.opti.constrained import BoxConstrainedQuadraticOptimizer


class ActiveSet(BoxConstrainedQuadraticOptimizer):
    # Apply the Active Set Method to the convex Box-Constrained Quadratic
    # program:
    #
    #  (P) min { 1/2 x^T Q x + q^T x : 0 <= x <= ub }
    #
    # - max_iter (integer scalar, optional, default value 1000): the maximum
    #   number of iterations
    #
    # Output:
    #
    # - v (real scalar): the best function value found so far (possibly the
    #   optimal one)
    #
    # - x ([n x 1] real column vector, optional): the best solution found so
    #   far (possibly the optimal one)
    #
    # - status (string, optional): a string describing the status of the
    #   algorithm at termination, with the following possible values:
    #
    #   = 'optimal': the algorithm terminated having proven that x is an
    #     (approximately) optimal solution, i.e., the norm of the gradient at x
    #     is less than the required threshold
    #
    #   = 'stopped': the algorithm terminated having exhausted the maximum
    #     number of iterations: x is the bast solution found so far, but not
    #     necessarily the optimal one

    def __init__(self,
                 quad,
                 ub,
                 x=None,
                 eps=1e-6,
                 tol=1e-8,
                 max_iter=1000,
                 callback=None,
                 callback_args=(),
                 verbose=False):
        super(ActiveSet, self).__init__(quad=quad,
                                        ub=ub,
                                        x=x,
                                        eps=eps,
                                        tol=tol,
                                        max_iter=max_iter,
                                        callback=callback,
                                        callback_args=callback_args,
                                        verbose=verbose)

    def minimize(self):

        self.f_x = self.f.function(self.x)

        # because all constraints are box ones, the active set is logically
        # partitioned onto the set of lower and upper bound constraints that are
        # active, L and U respectively. Of course, L and U have to be disjoint.
        # Since we start from the middle of the box, both the initial active sets
        # are empty
        L = np.full(self.f.ndim, False)  # indexes of variables fixed to the lower bound
        U = np.full(self.f.ndim, False)  # indexes of variables fixed to the upper bound

        # the set of "active variables", those that do *not* belong to any of the
        # two active sets and therefore are "free", is therefore the complement to
        # 1 : n of L union U; since L and U are empty now, A = 1 : n
        A = np.full(self.f.ndim, True)

        if self.verbose:
            print('iter\t cost\t\t|B|', end='')

        while True:
            if self.is_verbose():
                print('\n{:4d}\t{: 1.4e}\t{:d}\t'.format(self.iter, self.f_x, sum(L) + sum(U)), end='')

            try:
                self.callback()
            except StopIteration:
                break

            if self.iter >= self.max_iter:
                self.status = 'stopped'
                break

            # solve the unconstrained problem restricted to A the problem reads:
            #
            #  min { (1/2) x_A^T Q_{AA} x_A + (q_A + u_U^T Q_{UA}) x_A }
            #    [ + (1/2) x_U^T Q_{UU} x_U + q_U u_U ]
            #
            # and therefore the optimal solution is:
            #
            #   x_A* = -Q_{AA}^{-1} (q_A + u_U^T Q_{UA})
            #
            # not that this actually is a constrained problem subject to equality
            # constraints, but in our case equality constraints just fix variables
            # (and anyway, any QP problem with equality constraints reduces to an
            # unconstrained one)

            xs = np.zeros_like(self.x)
            xs[U] = self.ub[U]

            try:
                # use the Cholesky factorization to solve the linear system if Q_{AA} is
                # symmetric and positive definite, i.e., the function is strictly convex
                xs[A] = cho_solve(cho_factor(self.f.Q[A, :][:, A]),
                                  -(self.f.q[A] + self.f.Q[A, :][:, U].dot(self.ub[U])))
            except:  # np.linalg.LinAlgError:
                # since Q is is not strictly psd, i.e., the function is linear along the
                # eigenvectors correspondent to the null eigenvalues, the system has infinite
                # solutions, so we will choose the one that minimizes the 2-norm
                Q = self.f.Q[A, :][:, A]
                q = self.f.q[A] + self.f.Q[A, :][:, U].dot(self.ub[U])
                # `min ||Qx - q||` is formally equivalent to solve the linear system:
                #                       (Q^T Q) x = (Q^T q)^T x
                Q, q = np.inner(Q, Q), Q.T.dot(q)
                xs[A] = minres(Q, -q)[0]

            if np.logical_and(xs[A] <= self.ub[A] + 1e-12, xs[A] >= -1e-12).all():
                # the solution of the unconstrained problem is actually feasible

                # move the current point right there
                self.x = xs

                # compute function value and gradient
                self.f_x, self.g_x = self.f.function(self.x), self.f.jacobian(self.x)

                h = np.nonzero(np.logical_and(L, self.g_x < -1e-12))[0]
                if h.size > 0:
                    uppr = False
                else:
                    h = np.nonzero(np.logical_and(U, self.g_x > 1e-12))[0]
                    uppr = True

                if h.size == 0:

                    if self.f.ndim <= 3:
                        self.x0_history.append(self.x[0])
                        self.x1_history.append(self.x[1])
                        self.f_x_history.append(self.f_x)

                    self.status = 'optimal'
                    break

                else:
                    h = h[0]  # that's probably Bland's anti-cycle rule
                    A[h] = True
                    if uppr:
                        U[h] = False
                        if self.is_verbose():
                            print('\tI/O: O {:d}(U)'.format(h), end='')
                    else:
                        L[h] = False
                        if self.is_verbose():
                            print('\tI/O: O {:d}(L)'.format(h), end='')
            else:
                # the solution of the unconstrained problem is not feasible
                # this means that d = xs - x is a descent direction, use it
                # of course, only the "free" part really needs to be computed

                d = np.zeros_like(self.x)
                d[A] = xs[A] - self.x[A]

                # first, compute the maximum feasible step size max_t such that:
                #   0 <= x[i] + max_t * d[i] <= u[i]   for all i

                idx = np.logical_and(A, d > 0)  # positive gradient entries
                max_t = min((self.ub[idx] - self.x[idx]) / d[idx], default=np.inf)
                idx = np.logical_and(A, d < 0)  # negative gradient entries
                max_t = min(max_t, min(-self.x[idx] / d[idx], default=np.inf))

                # it is useless to compute the optimal t, because we know already
                # that it is 1, whereas max_t necessarily is < 1
                self.x += max_t * d

                # compute function value
                self.f_x = self.f.function(self.x)

                # update the active set(s)
                nL = np.logical_and(A, self.x <= 1e-12)
                L[nL] = True
                A[nL] = False

                nU = np.logical_and(A, self.x >= self.ub - 1e-12)
                U[nU] = True
                A[nU] = False

                if self.is_verbose():
                    print('\tI/O: I {:d}+{:d}'.format(sum(nL), sum(nU)), end='')

            try:
                self.check_lagrangian_dual_optimality()
            except StopIteration:
                break

            self.iter += 1

        self.check_lagrangian_dual_conditions()

        if self.verbose:
            print('\n')

        return self
