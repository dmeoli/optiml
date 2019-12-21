import matplotlib.pyplot as plt
import numpy as np

from functions import Rosenbrock, Ackley
from line_search import armijo_wolfe_line_search, backtracking_line_search


def NWTN(f, x, eps=1e-6, max_f_eval=1000, m1=0.01, m2=0.9, delta=1e-6, tau=0.9,
         sfgrd=0.01, m_inf=-np.inf, min_a=1e-12, verbose=False, plot=False):
    # Apply a classical Newton's method for the minimization of the provided
    # function f.
    #
    # - x is either a [ n x 1 ] real (column) vector denoting the input of
    #   f(), or [] (empty).
    #
    # Output:
    #
    # - v (real, scalar): if x == [] this is the best known lower bound on
    #   the unconstrained global optimum of f(); it can be -np.inf if either f()
    #   is not bounded below, or no such information is available. If x ~= []
    #   then v = f(x).
    #
    # - g (real, [ n x 1 ] real vector): this also depends on x. if x == []
    #   this is the standard starting point from which the algorithm should
    #   start, otherwise it is the gradient of f() at x (or a subgradient if
    #   f() is not differentiable at x, which it should not be if you are
    #   applying the gradient method to it).
    #
    # The other [optional] input parameters are:
    #
    # - x (either [ n x 1 ] real vector or [], default []): starting point.
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
    # - delta (real scalar, optional, default value 1e-6): minimum positive
    #   value for the eigenvalues of the modified Hessian used to compute the
    #   Newton direction
    #
    # - tau (real scalar, optional, default value 0.9): scaling parameter for
    #   the line search. In the Armijo-Wolfe line search it is used in the
    #   first phase: if the derivative is not positive, then the step is
    #   divided by tau (which is < 1, hence it is increased). In the
    #   Backtracking line search, each time the step is multiplied by tau
    #   (hence it is decreased).
    #
    # - sfgrd (real scalar, optional, default value 0.01): safeguard parameter
    #   for the line search. to avoid numerical problems that can occur with
    #   the quadratic interpolation if the derivative at one endpoint is too
    #   large w.r.t. the one at the other (which leads to choosing a point
    #   extremely near to the other endpoint), a *safeguarded* version of
    #   interpolation is used whereby the new point is chosen in the interval
    #   [ as * ( 1 + sfgrd ) , am * ( 1 - sfgrd ) ], being [ as , am ] the
    #   current interval, whatever quadratic interpolation says. If you
    #   experience problems with the line search taking too many iterations to
    #   converge at "nasty" points, try to increase this
    #
    # - m_inf (real scalar, optional, default value -np.inf): if the algorithm
    #   determines a value for f() <= m_inf this is taken as an indication that
    #   the problem is unbounded below and computation is stopped
    #   (a "finite -np.inf").
    #
    # - mina (real scalar, optional, default value 1e-16): if the algorithm
    #   determines a step size value <= mina, this is taken as an indication
    #   that something has gone wrong (the gradient is not a direction of
    #   descent, so maybe the function is not differentiable) and computation
    #   is stopped. It is legal to take mina = 0, thereby in fact skipping this
    #   test.
    #
    # Output:
    #
    # - x ([ n x 1 ] real column vector): the best solution found so far.
    #
    # - status (string): a string describing the status of the algorithm at
    #   termination
    #
    #   = 'optimal': the algorithm terminated having proven that x is a(n
    #     approximately) optimal solution, i.e., the norm of the gradient at x
    #     is less than the required threshold
    #
    #   = 'unbounded': the algorithm has determined an extremely large negative
    #     value for f() that is taken as an indication that the problem is
    #     unbounded below (a "finite -np.inf", see m_inf above)
    #
    #   = 'stopped': the algorithm terminated having exhausted the maximum
    #     number of iterations: x is the bast solution found so far, but not
    #     necessarily the optimal one
    #
    #   = 'error': the algorithm found a numerical error that prevents it from
    #     continuing optimization (see mina above)

    x = np.asarray(x)

    # reading and checking input
    if not np.isrealobj(x):
        return ValueError('x not a real vector')

    if x.shape[1] != 1:
        return ValueError('x is not a (column) vector')

    f_star = f.function([])

    n = x.shape[0]

    if not np.isreal(eps) or not np.isscalar(eps):
        return ValueError('eps is not a real scalar')

    if not np.isscalar(max_f_eval):
        return ValueError('max_f_eval is not an integer scalar')

    if not np.isscalar(m1):
        return ValueError('m1 is not a real scalar')

    if m1 <= 0 or m1 >= 1:
        return ValueError('m1 is not in (0,1)')

    if not np.isscalar(m1):
        return ValueError('m2 is not a real scalar')

    if not np.isscalar(delta):
        return ValueError('delta is not a real scalar')

    if delta < 0:
        return ValueError('delta must be > 0')

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
        return ValueError('mina is not a real scalar')

    if min_a < 0:
        return ValueError('mina is < 0')

    last_x = np.zeros((n, 1))  # last point visited in the line search
    last_g = np.zeros((n, 1))  # gradient of last_x
    last_h = np.zeros((n, n))  # Hessian of last_x
    f_eval = 1  # f() evaluations count ("common" with LSs)

    # initializations
    if f_star > -np.inf:
        if verbose:
            print('f_eval\trel gap\t\t|| g(x) ||\trate\t\tdelta\t', end='')
        prev_v = np.inf
    else:
        if verbose:
            print('f_eval\tf(x)\t\t\t|| g(x) ||\tdelta\t', end='')
    if verbose:
        print('\tls\tit\ta*')

    v, g, h = f.function(x), f.jacobian(x), f.hessian(x)
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
                print('{:4d}\t{:1.4e}\t\t{:1.4e}'.format(f_eval, v, ng), end='')

        # stopping criteria
        if ng <= eps * ng0:
            status = 'optimal'
            break

        if f_eval > max_f_eval:
            status = 'stopped'
            break

        # compute Newton's direction
        lambda_n = min(np.linalg.eigvalsh(h))  # smallest eigenvalue
        if lambda_n < delta:
            if verbose:
                print('\t{:1.4e}'.format(delta - lambda_n), end='')
            h = h + (delta - lambda_n) * np.eye(n)
        else:
            if verbose:
                print('\t{:1.4e}'.format(0), end='')

        d = -np.linalg.inv(h).dot(g)  # or np.linalg.solve(h, g)

        phi_p0 = g.T.dot(d).item()

        # compute step size: in Newton's method, the default initial step size is 1
        if 0 < m2 < 1:
            a, v, last_x, last_g, last_h, f_eval = armijo_wolfe_line_search(
                f, d, x, last_x, last_g, last_h, f_eval, max_f_eval, min_a, sfgrd, v, phi_p0, 1, m1, m2, tau, verbose)
        else:
            a, v, last_x, last_g, last_h, f_eval = backtracking_line_search(
                f, d, x, last_x, last_g, last_h, f_eval, max_f_eval, min_a, v, phi_p0, 1, m1, tau, verbose)

        # output statistics
        if verbose:
            print('\t{:1.4e}'.format(a))

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

        # update gradient and Hessian
        g = last_g
        h = last_h
        ng = np.linalg.norm(g)

    if verbose:
        print()
    if plot and n == 2:
        plt.show()
    return x, status


if __name__ == "__main__":
    print(NWTN(Ackley(), [[-1], [1]], verbose=True, plot=True))
