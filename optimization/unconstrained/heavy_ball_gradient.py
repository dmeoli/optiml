import matplotlib.pyplot as plt
import numpy as np

from optimization import Rosenbrock
from optimization.optimizer import LineSearchOptimizer
from optimization.unconstrained.line_search import armijo_wolfe_line_search, backtracking_line_search


class HeavyBallGradient(LineSearchOptimizer):
    # Apply a Heavy Ball Gradient approach for the minimization of the
    # provided function f.
    #
    # Input:
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
    # - beta (real scalar, optional, default value 0.9): if beta > 0 then it
    #   is taken as the fixed momentum term. If beta < 0, then abs(beta) is
    #   taken as the scaled momentum term, i.e.,
    #
    #        beta^i = abs(beta) * || g^i || / || x^i - x^{i - 1} ||
    #
    #   in such a way that beta near to 1 has a "significant impact"
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
    # - m1 (real scalar, optional, default value 0.01): parameter of the
    #   Armijo condition (sufficient decrease) in the line search . Has to be
    #   in (0,1)
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
    #   the Backtracking line search, each time the step is multiplied by tau
    #   (hence it is decreased).
    #
    # - sfgrd (real scalar, optional, default value 0.01): safeguard parameter
    #   for the line search. to avoid numerical problems that can occur with
    #   the quadratic interpolation if the derivative at one endpoint is too
    #   large w.r.t. the one at the other (which leads to choosing a point
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
    #   descent, so maybe the function is not differentiable) and the line
    #   search is stopped (but the algorithm as a whole is not, as it is a
    #   non-monotone algorithm).
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

    def __init__(self, f, x=None, beta=0.9, eps=1e-6, max_f_eval=1000, m1=0.01, m2=0.9, a_start=1,
                 tau=0.9, sfgrd=0.01, m_inf=-np.inf, min_a=1e-16, verbose=False, plot=False):
        super().__init__(f, x, eps, max_f_eval, m1, m2, a_start, tau, sfgrd, m_inf, min_a, verbose, plot)
        if not np.isscalar(beta):
            raise ValueError('beta is not a real scalar')
        self.beta = beta

    def minimize(self):
        f_star = self.f.function([])

        last_x = np.zeros((self.n,))  # last point visited in the line search
        last_g = np.zeros((self.n,))  # gradient of last_x
        f_eval = 1  # f() evaluations count ("common" with LSs)

        if self.verbose:
            if f_star > -np.inf:
                print('f_eval\trel gap', end='')
            else:
                print('f_eval\tf(x)', end='')
            print('\t\t|| g(x) ||\tls fev\ta*')

        v, g = self.f.function(self.x), self.f.jacobian(self.x)
        ng = np.linalg.norm(g)
        if self.eps < 0:
            ng0 = -ng  # norm of first subgradient
        else:
            ng0 = 1  # un-scaled stopping criterion

        past_d = np.zeros((self.n,))

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        while True:
            # output statistics
            if self.verbose:
                if f_star > -np.inf:
                    print('{:4d}\t{:1.4e}\t{:1.4e}'.format(f_eval, (v - f_star) / max(abs(f_star), 1), ng), end='')
                else:
                    print('{:4d}\t{:1.8e}\t\t{:1.4e}'.format(f_eval, v, ng), end='')

            # stopping criteria
            if ng <= self.eps * ng0:
                status = 'optimal'
                break

            if f_eval > self.max_f_eval:
                status = 'stopped'
                break

            # compute deflected gradient direction
            if f_eval == 1:
                d = -g
            else:
                if self.beta > 0:
                    beta_i = self.beta
                else:
                    beta_i = -self.beta * ng / np.linalg.norm(past_d)
                d = -g + beta_i * past_d

            phi_p0 = g.T * d

            # compute step size
            if 0 < self.m2 < 1:
                a, v, last_x, last_g, _, f_eval = \
                    armijo_wolfe_line_search(self.f, d, self.x, last_x, last_g, None, f_eval, self.max_f_eval,
                                             self.min_a, self.sfgrd, v, phi_p0, self.a_start, self.m1, self.m2,
                                             self.tau, self.verbose)
            else:
                a, v, last_x, last_g, _, f_eval = \
                    backtracking_line_search(self.f, d, self.x, last_x, last_g, None, f_eval, self.max_f_eval,
                                             self.min_a, v, phi_p0, self.a_start, self.m1, self.tau, self.verbose)

            # output statistics
            if self.verbose:
                print('\t{:1.2e}'.format(a))

            if a <= self.min_a:
                status = 'error'
                break

            if v <= self.m_inf:
                status = 'unbounded'
                break

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((self.x, last_x))
                contour_axes.plot(p_xy[:, 0], p_xy[:, 1], color='k')

            past_d = last_x - self.x
            self.x = last_x

            g = last_g
            ng = np.linalg.norm(g)

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.x, status


if __name__ == "__main__":
    print(HeavyBallGradient(Rosenbrock(), verbose=True, plot=True).minimize())
