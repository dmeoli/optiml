import matplotlib.pyplot as plt
import numpy as np

from functions import Rosenbrock, gen_quad_1
from line_search import armijo_wolfe_line_search, backtracking_line_search


def SDQ(f, x, f_star=np.inf, eps=1e-6, max_iter=1000, verbose=False, plot=False):
    """
    Apply the Steepest Descent algorithm with exact line search to the quadratic function.

        f(x) = 1/2 x^T Q x + q x

    :param Q:        ([n x n] real symmetric matrix, not necessarily positive
                     semidefinite): the Hessian (quadratic part) of f.
    :param q:        ([n x 1] real column vector): the linear part of f.
    :param x:        ([n x 1] real column vector): the point where to start the algorithm from.
    :param f_star:   (real scalar, optional, default value inf): optimal value of f.
                     If a non-inf value is provided it is used to print out statistics about
                     the convergence speed of the algorithm.
    :param eps:      (real scalar, optional, default value 1e-6): the accuracy in the stopping
                     criterion: the algorithm is stopped when the norm of the gradient is less
                     than or equal to eps.
    :param max_iter: (integer scalar, optional, default value 1000): the maximum number of iterations
    :param verbose:
    :param plot:
    :return x:       ([n x 1] real column vector): either the best solution found so far (possibly the
                     optimal one) or a direction proving the problem is unbounded below, depending on case
    :return status:  (string): a string describing the status of the algorithm at termination:
                     - 'optimal': the algorithm terminated having proven that x is a(n approximately) optimal
                     solution, i.e., the norm of the gradient at x is less than the required threshold;
                     - 'unbounded': the algorithm terminated having proven that the problem is unbounded below:
                     x contains a direction along which f is decreasing to -inf, either because f is linear
                     along x and the directional derivative is not zero, or because x is a direction with
                     negative curvature;
                     - 'stopped': the algorithm terminated having exhausted the maximum number of iterations:
                     x is the best solution found so far, but not necessarily the optimal one.
    """

    x = np.asarray(x)

    n = x.shape[0]

    if not np.isrealobj(x):
        return ValueError('x not a real vector')

    if x.shape[1] != 1:
        return ValueError('x is not a (column) vector')

    if x.size != f.hessian().shape[0]:
        return ValueError('x size does not match with Q')

    if not np.isrealobj(f_star) or not np.isscalar(f_star):
        return ValueError('f_star is not a real scalar')

    if not np.isrealobj(eps) or not np.isscalar(eps):
        return ValueError('eps is not a real scalar')

    if eps < 0:
        return ValueError('eps can not be negative')

    if not np.isscalar(max_iter):
        return ValueError('max_iter is not an integer scalar')

    if verbose:
        print('iter\tf(x)\t\t\t||nabla f(x)||', end='')
    if f_star < np.inf:
        if verbose:
            print('\tf(x) - f*\trate', end='')
        prev_v = np.inf
    if verbose:
        print()

    if plot and n == 2:
        surface_plot, contour_plot, contour_plot, contour_axes = f.plot()

    i = 1
    while True:
        # compute function value and gradient
        v, g = f.function(x), f.jacobian(x)
        ng = np.linalg.norm(g)

        # output statistics
        if verbose:
            print('{:4d}\t{:1.8e}\t\t{:1.4e}'.format(i, v, ng), end='')
        if f_star < np.inf:
            if verbose:
                print('\t{:1.4e}'.format(v - f_star), end='')
            if prev_v < np.inf:
                if verbose:
                    print('\t{:1.4e}'.format((v - f_star) / (prev_v - f_star)), end='')
            prev_v = v
        if verbose:
            print()

        # stopping criteria
        if ng <= eps:
            status = 'optimal'
            break

        if i > max_iter:
            status = 'stopped'
            break

        # check if f is unbounded below
        den = g.T.dot(f.hessian()).dot(g)

        if den <= 1e-12:
            # this is actually two different cases:
            #
            # - g.T.dot(Q).dot(g) = 0, i.e., f is linear along g, and since the
            #   gradient is not zero, it is unbounded below;
            #
            # - g.T.dot(Q).dot(g) < 0, i.e., g is a direction of negative curvature
            #   for f, which is then necessarily unbounded below.
            status = 'unbounded'
            break

        # compute step size
        a = ng ** 2 / den

        # plot the trajectory
        if plot and n == 2:
            p_xy = np.hstack((x, x + a * -g))
            contour_axes.plot(p_xy[0], p_xy[1], color='k')

        # <\nabla f(x_i), \nabla f(x_i+1)> = 0
        # assert np.isclose(f.jacobian(x).T.dot(f.jacobian(x + a * -d)), 0)

        # compute new point
        x = x + a * -g
        i += 1

    if verbose:
        print()
    if plot and n == 2:
        plt.show()
    return x, status


def SDG(f, x, eps=1e-6, max_f_eval=1000, m1=0.01, m2=0.9, a_start=1,
        tau=0.9, sfgrd=0.01, m_inf=-np.inf, min_a=1e-16, verbose=False, plot=False):
    """
    Apply the classical Steepest Descent algorithm for the minimization of
    the provided function f.
    # - x is either a [n x 1] real (column) vector denoting the input of
    #   f(), or [] (empty).
    #
    # - x (either [n x 1] real vector or [], default []): starting point.
    #   If x == [], the default starting point provided by f() is used.
    #
    # - eps (real scalar, optional, default value 1e-6): the accuracy in the
    #   stopping criterion: the algorithm is stopped when the norm of the
    #   gradient is less than or equal to eps. If a negative value is provided,
    #   this is used in a *relative* stopping criterion: the algorithm is
    #   stopped when the norm of the gradient is less than or equal to
    #   (- eps) * || norm of the first gradient ||.
    #
    # - max_f_eval (integer scalar, optional, default value 1000): the maximum
    #   number of function evaluations (hence, iterations will be not more than
    #   max_f_eval because at each iteration at least a function evaluation is
    #   performed, possibly more due to the line search).
    #
    # - m1 (real scalar, optional, default value 0.01): first parameter of the
    #   Armijo-Wolfe-type line search (sufficient decrease). Has to be in (0,1)
    #
    # - m2 (real scalar, optional, default value 0.9): typically the second
    #   parameter of the Armijo-Wolfe-type line search (strong curvature
    #   condition). It should to be in (0,1); if not, it is taken to mean that
    #   the simpler Backtracking line search should be used instead
    #
    # - a_start (real scalar, optional, default value 1): starting value of
    #   alpha in the line search (> 0)
    #
    # - tau (real scalar, optional, default value 0.9): scaling parameter for
    #   the line search. In the Armijo-Wolfe line search it is used in the
    #   first phase: if the derivative is not positive, then the step is
    #   divided by tau (which is < 1, hence it is increased). In the
    #   Backtracking line search, each time the step is multiplied by tau
    #   (hence it is decreased).
    #
    # - sfgrd (real scalar, optional, default value 0.01): safeguard parameter
    #   for the line search. To avoid numerical problems that can occur with
    #   the quadratic interpolation if the derivative at one endpoint is too
    #   large w.r.t. The one at the other (which leads to choosing a point
    #   extremely near to the other endpoint), a *safeguarded* version of
    #   interpolation is used whereby the new point is chosen in the interval
    #   [as * (1 + sfgrd) , am * (1 - sfgrd)], being [as , am] the
    #   current interval, whatever quadratic interpolation says. If you
    #   experience problems with the line search taking too many iterations to
    #   converge at "nasty" points, try to increase this
    #
    # - m_inf (real scalar, optional, default value -inf): if the algorithm
    #   determines a value for f() <= m_inf this is taken as an indication that
    #   the problem is unbounded below and computation is stopped
    #   (a "finite -inf").
    #
    # - min_a (real scalar, optional, default value 1e-16): if the algorithm
    #   determines a step size value <= min_a, this is taken as an indication
    #   that something has gone wrong (the gradient is not a direction of
    #   descent, so maybe the function is not differentiable) and computation
    #   is stopped. It is legal to take min_a = 0, thereby in fact skipping this
    #   test.
    #
    :return x: ([n x 1] real column vector): the best solution found so far.
    # - v (real, scalar): if x == [] this is the best known lower bound on
    #   the unconstrained global optimum of f(); it can be -inf if either f()
    #   is not bounded below, or no such information is available. If x ~= []
    #   then v = f(x).
    #
    # - g (real, [n x 1] real vector): this also depends on x. If x == []
    #   this is the standard starting point from which the algorithm should
    #   start, otherwise it is the gradient of f() at x (or a subgradient if
    #   f() is not differentiable at x, which it should not be if you are
    #   applying the gradient method to it).
    :return status: (string): a string describing the status of the algorithm at termination:
                    - 'optimal': the algorithm terminated having proven that x is a(n
                    approximately) optimal solution, i.e., the norm of the gradient at x
                    is less than the required threshold;
                    - 'unbounded': the algorithm has determined an extremely large negative
                    value for f() that is taken as an indication that the problem is
                    unbounded below (a "finite -inf", see m_inf above);
                    - 'stopped': the algorithm terminated having exhausted the maximum
                    number of iterations: x is the bast solution found so far, but not
                    necessarily the optimal one;
                    - 'error': the algorithm found a numerical error that prev_vents it from
                    continuing optimization (see min_a above).
    """

    x = np.asarray(x)

    # reading and checking input
    if not np.isrealobj(x):
        return ValueError('x not a real vector')

    if x.shape[1] != 1:
        return ValueError('x is not a (column) vector')

    f_star = f.function([])

    n = x.shape[0]

    if not np.isrealobj(eps) or not np.isscalar(eps):
        return ValueError('eps is not a real scalar')

    if not np.isscalar(max_f_eval):
        return ValueError('max_f_eval is not an integer scalar')

    if not np.isscalar(m1):
        return ValueError('m1 is not a real scalar')
    if m1 <= 0 or m1 >= 1:
        return ValueError('m1 is not in (0,1)')

    if not np.isscalar(m1):
        return ValueError('m2 is not a real scalar')

    if not np.isscalar(a_start):
        return ValueError('a_start is not a real scalar')

    if a_start < 0:
        return ValueError('a_start must be > 0')

    if not np.isscalar(tau):
        return ValueError('tau is not a real scalar')

    if tau <= 0 or tau >= 1:
        return ValueError('tau is not in (0,1)')

    if not np.isscalar(sfgrd):
        return ValueError('sfgrd is not a real scalar')

    if sfgrd <= 0 or sfgrd >= 1:
        return ValueError('sfgrd is not in (0,1)')

    if not np.isscalar(m_inf):
        return ValueError('m_inf is not a real scalar')

    if not np.isscalar(min_a):
        return ValueError('min_a is not a real scalar')

    if min_a < 0:
        return ValueError('min_a is < 0')

    last_x = np.zeros((n, 1))  # last point visited in the line search
    last_g = np.zeros((n, 1))  # gradient of last_x
    f_eval = 1  # f() evaluations count ("common" with LSs)

    if f_star > -np.inf:
        if verbose:
            print('f_eval\trel gap\t\t|| g(x) ||\t\trate\t', end='')
        prev_v = np.inf
    else:
        if verbose:
            print('f_eval\tf(x)\t\t\t|| g(x) ||\t', end='')
    if verbose:
        print('ls f_eval\ta*')

    v, g = f.function(x), f.jacobian(x)
    ng = np.linalg.norm(g)
    if eps < 0:
        ng0 = -ng  # norm of first subgradient
    else:
        ng0 = 1  # un-scaled stopping criterion

    if plot and n == 2:
        surface_plot, contour_plot, contour_plot, contour_axes = f.plot()

    while True:
        # output statistics
        if f_star > -np.inf:
            if verbose:
                print('{:4d}\t{:1.4e}\t{:1.4e}'.format(f_eval, (v - f_star) / max([abs(f_star), 1]), ng), end='')
            if prev_v < np.inf:
                if verbose:
                    print('\t{:1.4e}'.format((v - f_star) / (prev_v - f_star)), end='')
            else:
                if verbose:
                    print('\t\t\t', end='')
            prev_v = v
        else:
            if verbose:
                print('{:4d}\t{:1.8e}\t\t{:1.4e}'.format(f_eval, v, ng), end='')

        # stopping criteria
        if ng <= eps * ng0:
            status = 'optimal'
            break

        if f_eval > max_f_eval:
            status = 'stopped'
            break

        d = -g

        phi_p0 = -ng * ng

        # compute step size
        if 0 < m2 < 1:
            a, v, last_x, last_g, _, f_eval = \
                armijo_wolfe_line_search(f, d, x, last_x, last_g, None, f_eval, max_f_eval,
                                         min_a, sfgrd, v, phi_p0, a_start, m1, m2, tau, verbose)
        else:
            a, v, last_x, last_g, _, f_eval = \
                backtracking_line_search(f, d, x, last_x, last_g, None, f_eval, max_f_eval,
                                         min_a, v, phi_p0, a_start, m1, tau, verbose)

        # output statistics
        if verbose:
            print('\t\t{:1.4e}'.format(a))

        if a <= min_a:
            status = 'error'
            break

        if v <= m_inf:
            status = 'unbounded'
            break

        # plot the trajectory
        if plot and n == 2:
            p_xy = np.hstack((x, last_x))
            contour_axes.plot(p_xy[0], p_xy[1], color='k')

        # compute new point
        x = last_x

        # update gradient
        g = last_g
        ng = np.linalg.norm(g)

    if verbose:
        print()
    if plot and n == 2:
        plt.show()
    return x, status


if __name__ == "__main__":
    print(SDQ(gen_quad_1, [[-1], [1]], verbose=True, plot=True))
    print()
    print(SDG(Rosenbrock(), [[-1], [1]], verbose=True, plot=True))
