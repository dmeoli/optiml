import matplotlib.pyplot as plt
import numpy as np

from ml.initializers import RandomUniform
from optimization.optimizer import LineSearchOptimizer


class BFGS(LineSearchOptimizer):
    # Apply a Quasi-Newton approach, in particular using the celebrated
    # Broyden-Fletcher-Goldfarb-Shanno (BFGS) formula, for the minimization of
    # the provided function f.
    #
    # - x is either a [n x 1] real (column) vector denoting the input of
    #   f(), or [] (empty).
    #
    # Output:
    #
    # - v (real, scalar): if x == [] this is the best known lower bound on
    #   the unconstrained global optimum of f(); it can be -inf if either f()
    #   is not bounded below, or no such information is available. If x ~= []
    #   then v = f(x).
    #
    # - g (real, [n x 1] real vector): this also depends on x. if x == []
    #   this is the standard starting point from which the algorithm should
    #   start, otherwise it is the gradient of f() at x (or a subgradient if
    #   f() is not differentiable at x, which it should not be if you are
    #   applying the gradient method to it).
    #
    # The other [optional] input parameters are:
    #
    # - x (either [n x 1] real vector or [], default []): starting point.
    #   If x == [], the default starting point provided by f() is used.
    #
    # - delta (real scalar, optional, default value 1): the initial
    #   approximation of the Hessian is taken as delta * I if delta > 0;
    #   otherwise, the initial Hessian is approximated by finite differences
    #   with - delta as the step, and inverted just the once.
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
    #   [as * (1 + sfgrd), am * (1 - sfgrd)], being [as, am] the
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
    # Output:
    #
    # - x ([n x 1] real column vector): the best solution found so far.
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
    #     unbounded below (a "finite -inf", see m_inf above)
    #
    #   = 'stopped': the algorithm terminated having exhausted the maximum
    #     number of iterations: x is the bast solution found so far, but not
    #     necessarily the optimal one
    #
    #   = 'error': the algorithm found a numerical error that prevents it from
    #     continuing optimization (see min_a above)

    def __init__(self, f, wrt=RandomUniform, batch_size=None, eps=1e-6, max_iter=1000, max_f_eval=1000, m1=0.01, m2=0.9,
                 a_start=1, delta=1, tau=0.9, sfgrd=0.01, m_inf=-np.inf, min_a=1e-16, verbose=False, plot=False):
        super().__init__(f, wrt, batch_size, eps, max_iter, max_f_eval, m1, m2,
                         a_start, tau, sfgrd, m_inf, min_a, verbose, plot)
        if not np.isscalar(delta):
            raise ValueError('delta is not a real scalar')
        self.delta = delta

    def minimize(self):
        last_wrt = np.zeros((self.n,))  # last point visited in the line search
        last_g = np.zeros((self.n,))  # gradient of last_wrt
        f_eval = 1  # f() evaluations count ("common" with LSs)

        # initializations
        if self.verbose:
            if self.f.f_star() > -np.inf:
                print('f eval\trel gap', end='')
            else:
                print('f eval\tf(x)', end='')
            print('\t\t||g(x)||\tls\tit\ta*\t\t\trho')

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        for args, kwargs in self.args:
            if self.iter == 1:
                v, g = self.f.function(self.wrt, *args, **kwargs), self.f.jacobian(self.wrt, *args, **kwargs)
                ng = np.linalg.norm(g)
                if self.eps < 0:
                    ng0 = -ng  # norm of first subgradient
                else:
                    ng0 = 1  # un-scaled stopping criterion

                if self.delta > 0:
                    # initial approximation of inverse of Hessian = scaled identity
                    B = self.delta * np.eye(self.n)
                else:
                    # initial approximation of inverse of Hessian computed by finite
                    # differences of gradient
                    small_step = max(-self.delta, 1e-8)
                    B = np.zeros((self.n, self.n))
                    for i in range(self.n):
                        xp = self.wrt
                        xp[i] = xp[i] + small_step
                        gp = self.f.jacobian(xp, *args, **kwargs)
                        B[i] = ((gp - g) / small_step).T
                    B = (B + B.T) / 2  # ensure it is symmetric
                    lambda_n = min(np.linalg.eigvalsh(B))  # smallest eigenvalue
                    if lambda_n < 1e-6:
                        B = B + (1e-6 - lambda_n) * np.eye(self.n)
                    B = np.linalg.inv(B)

            # output statistics
            if self.verbose:
                if self.f.f_star() > -np.inf:
                    print('{:4d}\t{:1.4e}\t{:1.4e}'.format(f_eval, (v - self.f.f_star()) /
                                                           max(abs(self.f.f_star()), 1), ng), end='')
                else:
                    print('{:4d}\t{:1.4e}\t{:1.4e}'.format(f_eval, v, ng), end='')

            # stopping criteria
            if ng <= self.eps * ng0:
                status = 'optimal'
                break

            if self.iter > self.max_iter or f_eval > self.line_search.max_f_eval:
                status = 'stopped'
                break

            # compute approximation to Newton's direction
            d = -B.dot(g)

            phi_p0 = g.T.dot(d)

            # compute step size: as in Newton's method, the default initial step size is 1
            a, v, last_wrt, last_g, f_eval = self.line_search.search(
                d, self.wrt, last_wrt, last_g, f_eval, v, phi_p0, args, kwargs)

            # output statistics
            if self.verbose:
                print('\t{:1.4e}'.format(a), end='')

            if a <= self.line_search.min_a:
                status = 'error'
                break

            if v <= self.m_inf:
                status = 'unbounded'
                break

            # update approximation of the Hessian using the BFGS formula
            s = (last_wrt - self.wrt).reshape((self.n, 1))  # s^i = x^{i + 1} - x^i
            y = (last_g - g).reshape((self.n, 1))  # y^i = \nabla f(x^{i + 1}) - \nabla f(x^i)

            rho = y.T.dot(s).item()
            if rho < 1e-16:
                if self.verbose:
                    print('\nError: y^i s^i = {:1.4e}'.format(rho))
                status = 'error'
                break

            rho = 1 / rho

            if self.verbose:
                print('\t{:1.4e}'.format(rho))

            D = B.dot(y) * s.T
            B = B + rho * ((1 + rho * y.T.dot(B).dot(y)) * (s.dot(s.T)) - D - D.T)

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((self.wrt, last_wrt)).T
                contour_axes.quiver(p_xy[0, :-1], p_xy[1, :-1], p_xy[0, 1:] - p_xy[0, :-1], p_xy[1, 1:] - p_xy[1, :-1],
                                    scale_units='xy', angles='xy', scale=1, color='k')

            # update new point
            self.wrt[:] = last_wrt

            # update gradient
            g = last_g
            ng = np.linalg.norm(g)

            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, status
