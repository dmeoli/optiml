import matplotlib.pyplot as plt
import numpy as np

from ml.initializers import random_uniform
from optimization.optimizer import LineSearchOptimizer


class Subgradient(LineSearchOptimizer):
    # Apply the classical Subgradient Method for the minimization of the
    # provided function f.
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
    #   start, otherwise it is a subgradient of f() at x (possibly the
    #   gradient, but you should not apply this algorithm to a differentiable
    #   f)
    #
    # The other [optional] input parameters are:
    #
    # - x (either [n x 1] real vector or [], default []): starting point.
    #   If x == [], the default starting point provided by f() is used.
    #
    # - eps (real scalar, optional, default value 1e-6): the accuracy in the
    #   stopping criterion. If eps > 0, then a target-level Polyak step size
    #   with non-vanishing threshold is used, and eps is taken as the minimum
    #   *relative* value for the displacement, i.e.,
    #
    #       delta^i >= eps * max(abs(f(x^i)), 1)
    #
    #   is used as the minimum value for the displacement. If eps < 0 and
    #   v_* = f([]) > -inf, then the algorithm "cheats" and it does an
    #   *exact* Polyak step size with termination criteria
    #
    #       (f^i_{ref} - v_*) <= (- eps) * max(abs(v_*), 1)
    #
    #   Finally, if eps == 0 the algorithm rather uses a DSS (diminishing
    #   square-summable) step size, i.e., a_start * (1 / i) [see below]
    #
    # - a_start (real scalar, optional, default value 1e-4): if eps > 0, i.e.,
    #   a target-level Polyak step size with non-vanishing threshold is used,
    #   then a_start is used as the relative value to which the displacement is
    #   reset each time f(x^{i + 1}) <= f^i_{ref} - delta^i, i.e.,
    #
    #     delta^{i + 1} = a_start * max(abs(f^{i + 1}_{ref}), 1)
    #
    #   If eps == 0, i.e. a diminishing square-summable) step size is used, then
    #   a_start is used as the fixed scaling factor for the step size sequence
    #   a_start * (1 / i).
    #
    # - tau (real scalar, optional, default value 0.95): if eps > 0, i.e.,
    #   a target-level Polyak step size with non-vanishing threshold is used,
    #   then delta^{i + 1} = delta^i * tau each time
    #      f(x^{i + 1}) > f^i_{ref} - delta^i
    #
    # - max_f_eval (integer scalar, optional, default value 1000): the maximum
    #   number of function evaluations (hence, iterations, since there is
    #   exactly one function evaluation per iteration).
    #
    # - m_inf (real scalar, optional, default value -inf): if the algorithm
    #   determines a value for f() <= m_inf this is taken as an indication that
    #   the problem is unbounded below and computation is stopped
    #   (a "finite -inf").
    #
    # - min_a (real scalar, optional, default value 1e-16): if the algorithm
    #   determines a step size value <= min_a, this is taken as the fact that the
    #   algorithm has already obtained the most it can and computation is
    #   stopped. It is legal to take min_a = 0.
    #
    # Output:
    #
    # - x ([n x 1] real column vector): the best solution found so far.
    #
    # - status (string): a string describing the status of the algorithm at
    #   termination
    #
    #   = 'optimal': the algorithm terminated having proven that x is a(n
    #     approximately) optimal solution; this only happens when "cheating",
    #     i.e., explicitly uses v_* = f([]) > -inf, unless in the very
    #     unlikely case that f() spontaneously produces an almost-null
    #     subgradient
    #
    #   = 'unbounded': the algorithm has determined an extremely large negative
    #     value for f() that is taken as an indication that the problem is
    #     unbounded below (a "finite -inf", see m_inf above)
    #
    #   = 'stopped': the algorithm terminated having exhausted the maximum
    #     number of iterations: x is the bast solution found so far, but not
    #     necessarily the optimal one

    def __init__(self, f, wrt=random_uniform, batch_size=None, eps=1e-6, a_start=1., tau=0.95,
                 max_iter=1000, max_f_eval=1000, m_inf=-np.inf, min_a=1e-16, verbose=False, plot=False):
        super().__init__(f, wrt, batch_size, eps, max_iter, max_f_eval, a_start=a_start, tau=tau,
                         m_inf=m_inf, min_a=min_a, verbose=verbose, plot=plot)

    def minimize(self):

        if self.eps < 0 and self.f.f_star() == np.inf:
            # no way of cheating since the true optimal value is unknown
            self.eps = -self.eps  # revert to ordinary target level step size

        if self.verbose and not self.iter % self.verbose:
            print('iter\tf(x)\t\t||g(x)||', end='')
            if self.f.f_star() < np.inf:
                print('\tf(x) - f*\trate\t', end='')
                prev_v = np.inf
            print('\ta*', end='')

        x_ref = self.wrt
        f_ref = np.inf  # best f-value found so far
        if self.eps > 0:
            delta = 0  # required displacement from f_ref

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        for args in self.args:
            v, g = self.f.function(self.wrt, *args), self.f.jacobian(self.wrt, *args)
            ng = np.linalg.norm(g)

            if self.eps > 0:  # target-level step size
                if v <= f_ref - delta:  # found a "significantly" better point
                    delta = self.line_search.a_start * max(abs(v), 1)  # reset delta
                else:  # decrease delta
                    delta = max(delta * self.line_search.tau, self.eps * max(abs(min(v, f_ref)), 1))

            if v < f_ref:  # found a better f-value (however slightly better)
                f_ref = v  # update f_ref
                x_ref = self.wrt  # this is the incumbent solution

            # output statistics
            if self.verbose and not self.iter % self.verbose:
                print('\n{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, v, ng), end='')
                if self.f.f_star() < np.inf:
                    print('\t{:1.4e}'.format(v - self.f.f_star()), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((v - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    else:
                        print('\t\t\t', end='')
                    prev_v = v

            # stopping criteria
            if self.eps < 0 and f_ref - self.f.f_star() <= -self.eps * max(abs(self.f.f_star()), 1):
                x_ref = self.wrt
                status = 'optimal'
                break

            if ng < 1e-12:  # unlikely, but it could happen
                x_ref = self.wrt
                status = 'optimal'
                break

            if self.iter >= self.max_iter or self.iter > self.line_search.max_f_eval:
                status = 'stopped'
                break

            # compute step size
            if self.eps > 0:  # Polyak step size with target level
                a = (v - f_ref + delta) / ng
            elif self.eps < 0:  # true Polyak step size (cheating)
                a = (v - self.f.f_star()) / ng
            else:  # diminishing square-summable step size
                a = self.line_search.a_start * (1 / self.iter)

            # output statistics
            if self.verbose and not self.iter % self.verbose:
                print('\t{:1.4e}'.format(a), end='')

            # stopping criteria
            if a <= self.line_search.min_a:
                status = 'stopped'
                break

            if v <= self.m_inf:
                status = 'unbounded'
                break

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((self.wrt, self.wrt - (a / ng) * g)).T
                contour_axes.quiver(p_xy[0, :-1], p_xy[1, :-1], p_xy[0, 1:] - p_xy[0, :-1], p_xy[1, 1:] - p_xy[1, :-1],
                                    scale_units='xy', angles='xy', scale=1, color='k')

            # compute new point
            self.wrt = self.wrt - (a / ng) * g

            self.iter += 1

        x = x_ref  # return point corresponding to best value found so far

        if self.verbose and not self.iter % self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return x, v, status
