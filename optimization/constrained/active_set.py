from optimization.constrained.projected_gradient import ConstrainedOptimizer


class ActiveSet(ConstrainedOptimizer):
    # Apply the Active Set Method to the convex Box-Constrained Quadratic
    # program
    #
    #  (P) min { (1/2) x^T * Q * x + q * x : 0 <= x <= u }
    #
    # encoded in the structure BCQP, where Q must be strictly positive
    # definite.
    #
    # Input:
    #
    # - BCQP, the structure encoding the BCQP to be solved within its fields:
    #
    #   = BCQP.Q: n \times n symmetric positive semidefinite real matrix
    #
    #   = BCQP.q: n \times 1 real vector
    #
    #   = BCQP.u: n \times 1 real vector > 0
    #
    # - MaxIter (integer scalar, optional, default value 1000): the maximum
    #   number of iterations
    #
    # Output:
    #
    # - v (real scalar): the best function value found so far (possibly the
    #   optimal one)
    #
    # - x ([ n x 1 ] real column vector, optional): the best solution found so
    #   far (possibly the optimal one)
    #
    # - status (string, optional): a string describing the status of the
    #   algorithm at termination, with the following possible values:
    #
    #   = 'optimal': the algorithm terminated having proven that x is a(n
    #     approximately) optimal solution, i.e., the norm of the gradient at x
    #     is less than the required threshold
    #
    #   = 'stopped': the algorithm terminated having exhausted the maximum
    #     number of iterations: x is the bast solution found so far, but not
    #     necessarily the optimal one

    def __init__(self, f, eps=1e-6, max_iter=1000, verbose=False, plot=False):
        super().__init__(f, eps, max_iter, verbose, plot)

    def minimize(self, ub):

        x = BCQP.u / 2  # start from the middle of the box
        v = 0.5 * x.cT * BCQP.Q * x + BCQP.q.cT * x

        # Because all constraints are box ones, the active set is logically
        # partitioned onto the set of lower and upper bound constraints that are
        # active, L and U respectively. Of course, L and U have to be disjoint.
        # Since we start from the middle of the box, both the initial active sets
        # are empty
        L = false(n, 1)
        U = false(n, 1)

        # the set of "active variables", those that do *not* belong to any of the
        # two active sets and therefore are "free", is therefore the complement to
        # 1 : n of L union U; since L and U are empty now, A = 1 : n
        A = true(n, 1)

        fprintf(mstring('Active Set method\\n'))
        fprintf(mstring('iter\\tf(x)\\t\\t| B |\\tI/O\\n\\n'))

        i = 1

        # main loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        while True:
            # output statistics - - - - - - - - - - - - - - - - - - - - - - - - - - -

            fprintf(mstring('%4d\\t%1.8e\\t%d\\t'), i, v, sum(L) + sum(U))

            # stopping criteria - - - - - - - - - - - - - - - - - - - - - - - - - -

            if i > MaxIter:
                status = mstring('stopped')
                break

            # solve the *unconstrained* problem restricted to A - - - - - - - - - -
            # the problem reads
            #
            #  min { (1/2) x_A' * Q_{AA} * x_A + ( q_A + u_U' * Q_{UA} ) * x_A }
            #    [ + (1/2) x_U' * Q_{UU} * x_U ]
            #
            # and therefore the optimal solution is
            #
            #   x_A^* = - Q_{AA}^{-1} ( q_A + u_U' * Q_{UA} )
            #
            # not that this actually is a *constrained* problem subject to equality
            # constraints, but in our case equality constraints just fix variables
            # (and anyway, any QP problem with equality constraints reduces to an
            # unconstrained one)

            xs = zeros(n, 1)
            xs(U).lvalue = BCQP.u(U)
            opts.SYM = true  # tell it Q_{AA} is positive definite (hence of
            opts.POSDEF = true  # course symmetric), it'll probably do the right
            # thing and use Cholesky to solve the system
            xs(A).lvalue = linsolve(BCQP.Q(A, A), -(BCQP.q(A) + BCQP.Q(A, U) * BCQP.u(U)), opts)

            if all(xs(A) <= logical_and(BCQP.u(A) + 1e-12, xs(A) >= -1e-12)):
                # the solution of the unconstrained problem is actually feasible

                # move the current point right there
                x = xs

                # compute function value and gradient
                v = 0.5 * x.cT * BCQP.Q * x + BCQP.q.cT * x
                g = BCQP.Q * x + BCQP.q

                h = find(logical_and(L, g < -1e-12))
                if not isempty(h):
                    uppr = false
                else:
                    h = find(logical_and(U, g > 1e-12))
                    uppr = true

                if isempty(h):
                    fprintf(mstring('\\nOPT\\t%1.8e\\n'), v)
                    status = mstring('optimal')
                    break
                else:
                    h = h(1, 1)  # that's probably Bland's anti-cycle rule
                    A(h).lvalue = true
                    if uppr:
                        U(h).lvalue = false
                        fprintf(mstring('O %d(U)\\n'), h)
                    else:
                        L(h).lvalue = false
                        fprintf(mstring('O %d(L)\\n'), h)
            else:  # the solution of the unconstrained problem is *not* feasible
                # this means that d = xs - x is a descent direction, use it
                # of course, only the "free" part really needs to be computed

                d = zeros(n, 1)
                d(A).lvalue = xs(A) - x(A)

                # first, compute the maximum feasible stepsize maxt such that
                #
                #   0 <= x( i ) + maxt * d( i ) <= u( i )   for all i

                ind = logical_and(A, d > 0)  # positive gradient entries
                maxt = min((BCQP.u(ind) - x(ind)) / eldiv / d(ind))
                ind = logical_and(A, d < 0)  # negative gradient entries
                maxt = min(mcat([maxt, min(-x(ind) / eldiv / d(ind))]))

                # it is useless to compute the optimal t, because we know already
                # that it is 1, whereas maxt necessarily is < 1
                x = x + maxt * d

                # compute function value
                v = 0.5 * x.cT * BCQP.Q * x + BCQP.q.cT * x

                # update the active set(s)
                nL = logical_and(A, x <= 1e-12)
                L(nL).lvalue = true
                A(nL).lvalue = false

                nU = logical_and(A, x >= BCQP.u - 1e-12)
                U(nU).lvalue = true
                A(nU).lvalue = false

                fprintf(mstring('I %d+%d\\n'), sum(nL), sum(nU))

            # iterate - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            i = i + 1

        if nargout > 1:
            varargout(1).lvalue = x

        if nargout > 2:
            varargout(2).lvalue = status
