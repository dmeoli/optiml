from optimization.constrained.projected_gradient import ConstrainedOptimizer


class FrankWolfe(ConstrainedOptimizer):
    # function [ v , x , status ] = FWBCQP( BCQP , eps , MaxIter , t )
    #
    # Apply the (possibly, stabilized) Frank-Wolfe algorithm with exact line
    # search to the convex Box-Constrained Quadratic program
    #
    #  (P) min { (1/2) x^T * Q * x + q * x : 0 <= x <= u }
    #
    # encoded in the structure BCQP.
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
    # - eps (real scalar, optional, default value 1e-6): the accuracy in the
    #   stopping criterion: the algorithm is stopped when the relative gap
    #   between the value of the best primal solution (the current one) and the
    #   value of the best lower bound obtained so far is less than or equal to
    #   eps
    #
    # - MaxIter (integer scalar, optional, default value 1000): the maximum
    #   number of iterations
    #
    # - t (real scalar scalar, optional, default value 0): if the stablized
    #   version of the approach is used, then the new point is chosen in the
    #   box of relative sixe t around the current point, i.e., the component
    #   x[ i ] is allowed to change by not more than plus or minus t * u[ i ].
    #   if t = 0, then the non-stabilized version of the algorithm is used.
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

    if not isempty(varargin):
        eps = varargin(1)
        if not isreal(eps) or not isscalar(eps):
            error(mstring('eps is not a real scalar'))
        end
    else:
        eps = 1e-6
    end

    if length(varargin) > 1:
        MaxIter = round(varargin(2))
        if not isscalar(MaxIter):
            error(mstring('MaxIter is not an integer scalar'))
        end
    else:
        MaxIter = 1000
    end

    if length(varargin) > 2:
        t = varargin(3)
        if not isreal(t) or not isscalar(t):
            error(mstring('t is not a real scalar'))
        end
        if t < 0 or t > 1:
            error(mstring('t is not in [0, 1]'))
        end
    else:
        t = 0
    end

    # initializations - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    x = BCQP.u / 2  # start from the middle of the box

    bestlb = -inf  # best lower bound so far (= none, really)

    fprintf(mstring('Frank-Wolfe method\\n'))
    fprintf(mstring('iter\\tf(x)\\t\\tlb\\t\\tgap\\n\\n'))

    i = 1

    # main loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    while true:
        # compute function value and direction = - gradient - - - - - - - - - -

        v = 0.5 * x.cT * BCQP.Q * x + BCQP.q.cT * x
        g = BCQP.Q * x + BCQP.q

        # solve min { < g , y > : 0 <= y <= u }
        y = zeros(n, 1)
        ind = g < 0
        y(ind).lvalue = BCQP.u(ind)

        # compute the lower bound: remember that the first-order approximation
        # is f( x ) + g ( y - x )
        lb = v + g.cT * (y - x)
        if lb > bestlb:
            bestlb = lb
        end

        # compute the relative gap
        gap = (v - bestlb) / max(mcat([abs(v), 1]))

        # output statistics - - - - - - - - - - - - - - - - - - - - - - - - - - -

        fprintf(mstring('%4d\\t%1.8e\\t%1.8e\\t%1.4e\\n'), i, v, bestlb, gap)

        # stopping criteria - - - - - - - - - - - - - - - - - - - - - - - - - -
        if gap <= eps:
            status = mstring('optimal')
            break
        end

        if i > MaxIter:
            status = mstring('stopped')
            break
        end

        # in the stabilized case, restrict y in the box - - - - - - - - - - - -

        if t > 0:
            y = max(x - t * BCQP.u, min(x + t * BCQP.u, y))
        end

        # compute step size - - - - - - - - - - - - - - - - - - - - - - - - - -
        # we are taking direction d = y - x and y is feasible, hence the
        # maximum stepsize is 1

        d = y - x

        # compute optimal unbounded stepsize:
        # min (1/2) ( x + a d )' * Q * ( x + a d ) + q' * ( x + a d ) =
        #     (1/2) a^2 ( d' * Q * d ) + a d' * ( Q * x + q ) [ + const ]
        #
        # ==> a = - d' * ( Q * x + q ) / d' * Q * d
        #
        den = d.cT * BCQP.Q * d

        if den <= 1e-16:  # d' * Q * d = 0  ==>  f is linear along d
            alpha = 1  # just take the maximum possible stepsize
        else:
            # optimal unbounded stepsize restricted to max feasible step
            alpha = min(mcat([(-g.cT * d) / den, 1]))
        end

        # compute new point - - - - - - - - - - - - - - - - - - - - - - - - - -

        x = x + alpha * d

        # iterate - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        i = i + 1

    end

    # end of main loop- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if nargout > 1:
        varargout(1).lvalue = x
    end

    if nargout > 2:
        varargout(2).lvalue = status
    end


end  # the end- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
