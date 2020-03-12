from optimization.constrained.projected_gradient import ConstrainedOptimizer


class LagrangianDual(ConstrainedOptimizer):
    # Solve the convex Box-Constrained Quadratic program
    #
    #  (P) min { (1/2) x^T * Q * x + q * x : 0 <= x <= u }
    #
    # encoded in the structure BCQP using a dual approach, where Q must be
    # strictly positive definite. The box constraints 0 <= x <= u are relaxed
    # (with Lagrangian multipliers \lambda^- and \lambda^+, respectively) and
    # the corresponding Lagrangian Dual is solved by means of an ad-hoc
    # implementation of the Projected Gradient method (since the Lagrangian
    # multipliers are constrained in sign, and the Lagrangian function is
    # differentiable owing to strict-positive-definiteness of Q) using a
    # classical Armijo-Wolfe line search.
    #
    # A rough Lagrangian heuristic is implemented whereby the dual solution is
    # projected on the box at each iteration to provide an upper bound, which
    # is then used in the stopping criterion.
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
    # - dolh (logical scalar, optional, default value 1): true if a quick
    #   Lagrangian Heuristic is implemented whereby the current solution of the
    #   Lagrangian subproblem is projected on the box constraints
    #
    # - eps (real scalar, optional, default value 1e-6): the accuracy in the
    #   stopping criterion. This depends on the dolh parameter, see above: if
    #   dolh = true then the algorithm is stopped when the relative gap
    #   between the value of the best dual solution (the current one) and the
    #   value of the best upper bound obtained so far (by means of the
    #   heuristic) is less than or equal to eps, otherwise the stopping
    #   criterion is on the (relative, w.r.t. the first one) norm of the
    #   (projected) gradient
    #
    # - MaxFeval (integer scalar, optional, default value 1000): the maximum
    #   number of function evaluations (hence, iterations will be not more than
    #   MaxFeval because at each iteration at least a function evaluation is
    #   performed, possibly more due to the line search).
    #
    # - m1 (real scalar, optional, default value 0.01): first parameter of the
    #   Armijo-Wolfe-type line search (sufficient decrease). Has to be in (0,1)
    #
    # - m2 (real scalar, optional, default value 0.9): the second parameter of
    #   the Armijo-Wolfe-type line search (strong curvature condition), it
    #   should to be in (0,1);
    #
    # - astart (real scalar, optional, default value 1): starting value of
    #   alpha in the line search (> 0)
    #
    # - sfgrd (real scalar, optional, default value 0.01): safeguard parameter
    #   for the line search. to avoid numerical problems that can occur with
    #   the quadratic interpolation if the derivative at one endpoint is too
    #   large w.r.t. the one at the other (which leads to choosing a point
    #   extremely near to the other endpoint), a *safeguarded* version of
    #   interpolation is used whereby the new point is chosen in the interval
    #   [ as * ( 1 + sfgrd ) , am * ( 1 - sfgrd ) ], being [ as , am ] the
    #   current interval, whatever quadratic interpolation says. If you
    #   experiemce problems with the line search taking too many iterations to
    #   converge at "nasty" points, try to increase this
    #
    # - mina (real scalar, optional, default value 1e-16): if the algorithm
    #   determines a stepsize value <= mina, this is taken as an indication
    #   that something has gone wrong (the gradient is not a direction of
    #   descent, so maybe the function is not differentiable) and computation
    #   is stopped. It is legal to take mina = 0, thereby in fact skipping this
    #   test.
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
    #
    #   = 'error': the algorithm found a numerical error that prevents it from
    #     continuing optimization (see mina above)
    #
    # - l ([ 2 * n x 1 ] real column vector, optional): the best Lagrangian
    #   multipliers found so far (possibly the optimal ones)

    if not isstruct(BCQP):
        error(mstring('BCQP not a struct'))
    end

    if not isfield(BCQP, mstring('Q')) or not isfield(BCQP, mstring('q')) or not isfield(BCQP, mstring('u')):
        error(mstring('BCQP not a well-formed struct'))
    end

    if not isreal(BCQP.Q) or not isreal(BCQP.q) or not isreal(BCQP.u):
        error(mstring('BCQP not a well-formed struct'))
    end

    n = size(BCQP.q, 1)
    if size(BCQP.q, 2) != 1 or not isequal(size(BCQP.Q), mcat([n, n])) or not isequal(size(BCQP.u), mcat([n, 1])):
        error(mstring('BCQP not a well-formed struct'))
    end

    # compute straight away the Cholesky factorization of Q, this will be used
    # at each iteration to solve the Lagrangian relaxation
    [R, p] = chol(BCQP.Q)
    if p > 0:
        error(mstring('BCQP.Q not positive definite, this is not supported (yet)'))
    end

    if not isempty(varargin):
        dolh = logical(varargin(1))
    else:
        dolh = true
    end

    if length(varargin) > 1:
        eps = varargin(2)
        if not isreal(eps) or not isscalar(eps):
            error(mstring('eps is not a real scalar'))
        end
    else:
        eps = 1e-6
    end

    if length(varargin) > 2:
        MaxFeval = round(varargin(3))
        if not isscalar(MaxFeval):
            error(mstring('MaxFeval is not an integer scalar'))
        end
    else:
        MaxFeval = 1000
    end

    if length(varargin) > 3:
        m1 = varargin(4)
        if not isscalar(m1):
            error(mstring('m1 is not a real scalar'))
        end
        if m1 <= 0 or m1 >= 1:
            error(mstring('m1 is not in (0 ,1)'))
        end
    else:
        m1 = 0.01
    end

    if length(varargin) > 4:
        m2 = varargin(5)
        if not isscalar(m1):
            error(mstring('m2 is not a real scalar'))
        end
        if m2 <= 0 or m2 >= 1:
            error(mstring('m2 is not in (0, 1)'))
        end
    else:
        m2 = 0.9
    end

    if length(varargin) > 5:
        astart = varargin(6)
        if not isscalar(astart):
            error(mstring('astart is not a real scalar'))
        end
        if astart < 0:
            error(mstring('astart must be > 0'))
        end
    else:
        astart = 1
    end

    if length(varargin) > 6:
        sfgrd = varargin(7)
        if not isscalar(sfgrd):
            error(mstring('sfgrd is not a real scalar'))
        end
        if sfgrd <= 0 or sfgrd >= 1:
            error(mstring('sfgrd is not in (0, 1)'))
        end
    else:
        sfgrd = 0.01
    end

    if length(varargin) > 7:
        mina = varargin(8)
        if not isscalar(mina):
            error(mstring('mina is not a real scalar'))
        end
        if mina < 0:
            error(mstring('mina is < 0'))
        end
    else:
        mina = 1e-12
    end

    # initializations - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    x = BCQP.u / 2  # initial feasible solution is the middle of the box
    v = 0.5 * x.cT * BCQP.Q * x + BCQP.q.cT * x

    fprintf(mstring('Lagrangian Dual\\n'))
    if dolh:
        fprintf(mstring('feval\\tub\\t\\tp(l)\\t\\tgap\\t\\tls feval\\ta*\\n\\n'))
    else:
        fprintf(mstring('feval\\tp(l)\\t\\t|| grad ||\\tls feval\\ta*\\n\\n'))
    end

    feval = 0

    _lambda = zeros(2 * n, 1)
    [p, lastg] = phi(_lambda)

    # main loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    while true:
        # project the direction = - gradient over the active constraints- - - -
        d = -lastg
        d(_lambda <= logical_and(1e-12, d < 0)).lvalue = 0

        # output statistics - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if dolh:
            # compute the relative gap
            gap = (v + p) / max(mcat([abs(v), 1]))

            fprintf(mstring('%4d\\t%1.8e\\t%1.8e\\t%1.4e\\t'), feval, v, -p, gap)

            if gap <= eps:
                fprintf(mstring('OPT\\n'))
                status = mstring('optimal')
                break
            end
        else:
            # compute the norm of the projected gradient
            gnorm = norm(d)

            fprintf(mstring('%4d\\t%1.8e\\t%1.4e\\t'), feval, -p, gnorm)

            if feval == 1:
                gnorm0 = gnorm
            end
            if gnorm <= eps * gnorm0:
                fprintf(mstring('OPT\\n'))
                status = mstring('optimal')
                break
            end
        end
        # stopping criteria - - - - - - - - - - - - - - - - - - - - - - - - - -

        if feval > MaxFeval:
            fprintf(mstring('STOP\\n'))
            status = mstring('stopped')
            break
        end

        # compute step size - - - - - - - - - - - - - - - - - - - - - - - - - -

        # first, compute the maximum feasible stepsize maxt such that
        #
        #   0 <= lambda( i ) + maxt * d( i )   for all i

        ind = d < 0  # negative gradient entries
        if any(ind):
            maxt = min(astart, min(-_lambda(ind) / eldiv / d(ind)))
        else:
            maxt = astart
        end

        # now run the line search
        phip0 = lastg.cT * d
        [a, p] = ArmijoWolfeLS(p, phip0, maxt, m1, m2)

        fprintf(mstring('\\t%1.4e\\n'), a)

        if a <= mina:
            fprintf(mstring('\\tERR\\n'))
            status = mstring('error')
            break
        end

        # compute new point - - - - - - - - - - - - - - - - - - - - - - - - - -

        _lambda = _lambda + a * d

        # iterate - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    end

    # end of main loop- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if nargout > 1:
        varargout(1).lvalue = x
    end

    if nargout > 2:
        varargout(2).lvalue = status
    end

    if nargout > 3:
        varargout(3).lvalue = _lambda
    end

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # inner functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


@mfunction("p, y")
def solveLagrangian(_lambda=None):
    # The Lagrangian relaxation of the problem is
    #
    #  min { (1/2) x' * Q * x + q * x - lambda^+ * ( u - x ) - lambda^- * ( x )
    # = min { (1/2) x' * Q * x + ( q + lambda^+ - lambda^- ) * x - lambda^+ * u
    #
    # where lambda^+ are the first n components of lambda[], and lambda^- the
    # last n components; both are constrained to be >= 0
    #
    # The optimal solution of the Lagrangian relaxation is the (unique)
    # solution of the linear system
    #
    #       Q * x = - q - lambda^+ + lambda^-
    #
    # Since we have computed at the beginning the Cholesky factorization of Q,
    # i.e., Q = R' * R, where R is upper triangular and therefore RT is lower
    # triangular, we obtain this by just two triangular backsolves:
    #
    #       R' * z = - q - lambda^+ + lambda^-
    #
    #       R * x = z
    #
    # return the function value and the primal solution

    ql = BCQP.q + _lambda(mslice[1:n]) - _lambda(mslice[n + 1:end])
    opts.LT = true
    z = linsolve(R.cT, -ql, opts)
    opts.LT = false
    opts.UT = true
    y = linsolve(R, z, opts)

    # compute phi-value
    p = (0.5 * y.cT * BCQP.Q + ql.cT) * y - _lambda(mslice[1:n]).cT * BCQP.u

    feval = feval + 1


@mfunction("p, g")
def phi(_lambda=None):
    # phi( lambda ) is the Lagrangian function of the problem. With x the
    # optimal solution of the minimization problem (see solveLagrangian()), the
    # gradient at lambda is [ x - u ; - x ]
    #
    # however, the line search is written for minimization but we rather want
    # to maximize phi(), hence we have to change the sign of both function
    # values and gradient entries

    # solve the Lagrangian relaxation
    [p, y] = solveLagrangian(_lambda)
    p = -p

    # compute gradient
    g = mcat([BCQP.u - y, OMPCSEMI, y])

    if dolh:
        # compute an heuristic solution out of the solution y of the Lagrangian
        # relaxation by projecting y on the box

        y(y < 0).lvalue = 0
        ind = y > BCQP.u
        y(ind).lvalue = BCQP.u(ind)

        # compute cost of feasible solution
        pv = 0.5 * y.cT * BCQP.Q * y + BCQP.q.cT * y

        if pv < v:  # it is better than best one found so far
            x = y  # y becomes the incumbent
            v = pv


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def phi1d(alpha=None):
    # phi( lambda ) is the Lagrangian function of the problem; then,
    # phi( alpha ) = phi( lambda + alpha d ) and
    # phi'( alpha ) = < \nabla phi( lambda + alpha * d ) , d >

    [p, lastg] = phi(_lambda + alpha * d)
    pp = d.cT * lastg


def ArmijoWolfeLS(phi0=None, phip0=None, _as=None, m1=None, m2=None):
    # performs an Armijo-Wolfe Line Search.
    #
    # phi0 = phi( 0 ), phip0 = phi'( 0 ) < 0
    #
    # as > 0 is the first value to be tested, and it is also the *maximum*
    # possible stepsize: if phi'( as ) < 0 then the LS is immediately
    # terminated
    #
    # m1 and m2 are the standard Armijo-Wolfe parameters; note that the strong
    # Wolfe condition is used
    #
    # returns the optimal step and the optimal f-value

    a = _as
    [phia, phips] = phi1d(a)
    if phips <= 0:
        fprintf(mstring('%2d'), 1)
        return

    lsiter = 1  # count ls iterations

    am = 0
    phipm = phip0
    while (feval <= MaxFeval) and ((_as - am)) > mina and (phips > 1e-12):

        # compute the new value by safeguarded quadratic interpolation
        a = (am * phips - _as * phipm) / (phips - phipm)
        a = max(mcat([am * (1 + sfgrd), min(mcat([_as * (1 - sfgrd), a]))]))

        # compute phi( a )
        [phia, phip] = phi1d(a)

        if (phia <= phi0 + m1 * a * phip0) and (abs(phip) <= -m2 * phip0):
            break  # Armijo + strong Wolfe satisfied, we are done

        # restrict the interval based on sign of the derivative in a
        if phip < 0:
            am = a
            phipm = phip
        else:
            _as = a
            if _as <= mina:
                break
            phips = phip
        lsiter = lsiter + 1

    fprintf(mstring('%2d'), lsiter)
