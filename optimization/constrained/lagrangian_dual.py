import matplotlib.pyplot as plt
import numpy as np

from optimization.optimization_function import BoxConstrainedQuadratic, LagrangianBoxConstrained
from optimization.optimizer import LineSearchOptimizer


class LagrangianDual(LineSearchOptimizer):
    # Solve the convex Box-Constrained Quadratic program:
    #
    #  (P) min { 1/2 x^T Q x + q^T x : 0 <= x <= ub }
    #
    # The box constraints 0 <= x <= u are relaxed (with Lagrangian multipliers
    # \lambda^- and \lambda^+, respectively) and the corresponding Lagrangian
    # dual is solved by means of an ad-hoc implementation of the Projected
    # Gradient method (since the Lagrangian multipliers are constrained in
    # sign, and the Lagrangian function is differentiable owing to
    # strict-positive-definiteness of Q) using a classical Armijo Wolfe line search.
    #
    # A rough Lagrangian heuristic is implemented whereby the dual solution is
    # projected on the box at each iteration to provide an upper bound, which
    # is then used in the stopping criterion.
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
    # - max_f_eval (integer scalar, optional, default value 1000): the maximum
    #   number of function evaluations (hence, iterations will be not more than
    #   max_f_eval because at each iteration at least a function evaluation is
    #   performed, possibly more due to the line search).
    #
    # - m1 (real scalar, optional, default value 0.01): first parameter of the
    #   Armijo-Wolfe-type line search (sufficient decrease). Has to be in (0,1)
    #
    # - m2 (real scalar, optional, default value 0.9): the second parameter of
    #   the Armijo-Wolfe-type line search (strong curvature condition), it
    #   should to be in (0,1);
    #
    # - a_start (real scalar, optional, default value 1): starting value of
    #   alpha in the line search (> 0)
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
    # - min_a (real scalar, optional, default value 1e-16): if the algorithm
    #   determines a step size value <= mina, this is taken as an indication
    #   that something has gone wrong (the gradient is not a direction of
    #   descent, so maybe the function is not differentiable) and computation
    #   is stopped. It is legal to take min_a = 0, thereby in fact skipping this
    #   test.
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
    #
    #   = 'error': the algorithm found a numerical error that prevents it from
    #     continuing optimization (see mina above)
    #
    # - l ([2 * n x 1] real column vector, optional): the best Lagrangian
    #   multipliers found so far (possibly the optimal ones)

    def __init__(self, f, dolh=True, eps=1e-6, max_iter=1000, max_f_eval=1000, m1=0.01, m2=0.9, a_start=1,
                 tau=0.95, sfgrd=0.01, m_inf=-np.inf, min_a=1e-12, verbose=False, plot=False):
        if not isinstance(f, BoxConstrainedQuadratic):
            raise TypeError('f is not a box-constrained quadratic function')
        self.primal = f
        super().__init__(LagrangianBoxConstrained(f), f.ub / 2,  # start from the middle of the box,
                         eps=eps, max_iter=max_iter, max_f_eval=max_f_eval, m1=m1, m2=m2, a_start=a_start,
                         tau=tau, sfgrd=sfgrd, m_inf=m_inf, min_a=min_a, verbose=verbose, plot=plot)
        self.dolh = dolh

    def _dolh(self, v, last_g):
        if self.dolh:
            y = np.copy(last_g[self.primal.n:])

            # compute an heuristic solution out of the solution y of
            # the Lagrangian relaxation by projecting y on the box
            y[y < 0] = 0
            idx = y > self.primal.ub
            y[idx] = self.primal.ub[idx]

            # compute cost of feasible solution
            pv = self.primal.function(y)

            if pv < v:  # it is better than best one found so far
                self.wrt = np.copy(y)  # y becomes the incumbent
                v = pv

        return v

    def minimize(self):
        last_lmbda = np.zeros(2 * self.f.n)  # last point visited in the line search
        f_eval = 1  # f() evaluations count ("common" with LSs)

        v = self.primal.function(self.wrt)

        if self.verbose:
            print('iter\tf eval\t', end='')
            if self.dolh:
                print('ub\t\t\tf(l)\t\tgap\t\t', end='')
            else:
                print('f(l)\t\t\t||g(l)||', end='')
            print('\tls\tit\ta*')

        lmbda = np.zeros(2 * self.f.n)

        p, last_g = self.f.function(lmbda), self.f.jacobian(lmbda)
        v = self._dolh(v, last_g)

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        while True:
            # project the direction = -gradient over the active constraints
            d = -last_g
            d[np.logical_and(lmbda <= 1e-12, d < 0)] = 0

            if self.dolh:
                # compute the relative gap
                gap = (v + p) / max(abs(v), 1)

                if self.verbose:
                    print('{:4d}\t{:4d}\t{:1.4e}\t{:1.4e}\t{:1.4e}'.format(self.iter, f_eval, v, -p, gap), end='')

                if gap <= self.eps:
                    status = 'optimal'
                    break
            else:
                # compute the norm of the projected gradient
                ng = np.linalg.norm(d)

                if self.verbose:
                    print('{:4d}\t{:4d}\t{:1.8e}\t{:1.4e}'.format(self.iter, f_eval, -p, ng), end='')

                if self.iter is 1:
                    ng0 = ng
                if ng <= self.eps * ng0:
                    status = 'optimal'
                    break

            if f_eval >= self.line_search.max_f_eval:
                status = 'stopped'
                break

            # first, compute the maximum feasible step size max_t such that:
            #
            #   0 <= lambda[i] + max_t * d[i]   for all i

            idx = d < 0  # negative gradient entries
            if any(idx):
                max_t = min(self.line_search.a_start, min(-lmbda[idx] / d[idx]))
            else:
                max_t = self.line_search.a_start

            phi_p0 = last_g.T.dot(d)

            # compute step size
            self.line_search.a_start = max_t
            a, p, last_lmbda, last_g, f_eval = self.line_search.search(d, lmbda, last_lmbda, last_g, f_eval, p, phi_p0)
            v = self._dolh(v, last_g)

            if self.verbose:
                print('\t{:1.4e}'.format(a))

            if a <= self.line_search.min_a:
                status = 'error'
                break

            lmbda += a * d

            # TODO add plotting

            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, v, status, lmbda
