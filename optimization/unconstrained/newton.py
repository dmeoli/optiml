import matplotlib.pyplot as plt
import numpy as np

from optimization.optimizer import LineSearchOptimizer


class Newton(LineSearchOptimizer):
    # Apply a classical Newton's method for the minimization of the provided
    # function f.
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
    # - mina (real scalar, optional, default value 1e-16): if the algorithm
    #   determines a step size value <= mina, this is taken as an indication
    #   that something has gone wrong (the gradient is not a direction of
    #   descent, so maybe the function is not differentiable) and computation
    #   is stopped. It is legal to take mina = 0, thereby in fact skipping this
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
    #     continuing optimization (see mina above)

    def __init__(self, f, wrt=None, eps=1e-6, max_f_eval=1000, m1=0.01, m2=0.9, a_start=1, delta=1e-6, tau=0.9,
                 sfgrd=0.01, m_inf=-np.inf, min_a=1e-12, verbose=False, plot=False):
        super().__init__(f, wrt, eps, max_f_eval, m1, m2, a_start, tau, sfgrd, m_inf, min_a, verbose, plot)
        if not np.isscalar(delta):
            raise ValueError('delta is not a real scalar')
        if delta < 0:
            raise ValueError('delta must be > 0')
        self.delta = delta

    def minimize(self):
        f_star = self.f.function([])

        last_wrt = np.zeros((self.n,))  # last point visited in the line search
        last_g = np.zeros((self.n,))  # gradient of last_wrt
        last_h = np.zeros((self.n, self.n))  # Hessian of last_wrt
        f_eval = 1  # f() evaluations count ("common" with LSs)

        # initializations
        if self.verbose:
            if f_star > -np.inf:
                print('f_eval\trel gap\t\t|| g(x) ||\trate\t\tdelta\t', end='')
                prev_v = np.inf
            else:
                print('f_eval\tf(x)\t\t\t|| g(x) ||\tdelta\t', end='')
            print('\tls\tit\ta*')

        v, g, H = self.f.function(self.wrt), self.f.jacobian(self.wrt), self.f.hessian(self.wrt)
        ng = np.linalg.norm(g)
        if self.eps < 0:
            ng0 = -ng  # norm of first subgradient
        else:
            ng0 = 1  # un-scaled stopping criterion

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        while True:
            if self.verbose:
                # output statistics
                if f_star > -np.inf:
                    print('{:4d}\t{:1.4e}\t{:1.4e}'.format(f_eval, (v - f_star) / max(abs(f_star), 1), ng), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((v - f_star) / (prev_v - f_star)), end='')
                    else:
                        print('\t\t\t', end='')
                    prev_v = v
                else:
                    print('{:4d}\t{:1.4e}\t\t{:1.4e}'.format(f_eval, v, ng), end='')

            # stopping criteria
            if ng <= self.eps * ng0:
                status = 'optimal'
                break

            if f_eval > self.max_f_eval:
                status = 'stopped'
                break

            # compute Newton's direction
            lambda_n = min(np.linalg.eigvalsh(H))  # smallest eigenvalue
            if lambda_n < self.delta:
                if self.verbose:
                    print('\t{:1.4e}'.format(self.delta - lambda_n), end='')
                H = H + (self.delta - lambda_n) * np.eye(self.n)
            else:
                if self.verbose:
                    print('\t{:1.4e}'.format(0), end='')

            d = -np.linalg.inv(H).dot(g)  # or np.linalg.solve(H, g)

            phi_p0 = g.T.dot(d)

            # compute step size: in Newton's method, the default initial step size is 1
            a, v, last_wrt, last_g, f_eval = self.line_search.search(d, self.wrt, last_wrt, last_g, f_eval,
                                                                     self.a_start, v, phi_p0)

            # output statistics
            if self.verbose:
                print('\t{:1.4e}'.format(a))

            if a <= self.min_a:
                status = 'error'
                break

            if v <= self.m_inf:
                status = 'unbounded'
                break

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((self.wrt, last_wrt))
                contour_axes.plot(p_xy[:, 0], p_xy[:, 1], color='k')

            # update new point
            self.wrt = last_wrt

            # update gradient and Hessian
            g = last_g
            H = self.f.hessian(self.wrt)
            ng = np.linalg.norm(g)

            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, status


if __name__ == "__main__":
    import optimization.test_functions as tf

    print(Newton(tf.quad1, [-1, 1], verbose=True, plot=True).minimize())
