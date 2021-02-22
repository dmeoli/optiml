import numpy as np

from . import LineSearchOptimizer


class ConjugateGradient(LineSearchOptimizer):
    # Apply a Nonlinear Conjugated Gradient algorithm for the minimization of
    # the provided function f.
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
    # - wf (integer scalar, optional, default value 0): which of the Nonlinear
    #   Conjugated Gradient formulae to use. Possible values are:
    #   = fr: Fletcher-Reeves
    #   = pr: Polak-Ribière
    #   = hs: Hestenes-Stiefel
    #   = dy: Dai-Yuan
    #
    # - r_start (integer scalar, optional, default value 0): if > 0, restarts
    #   (setting beta = 0) are performed every n * r_start iterations
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

    def __init__(self,
                 f,
                 x=None,
                 wf='fr',
                 eps=1e-6,
                 max_iter=1000,
                 max_f_eval=1000,
                 r_start=0,
                 m1=0.01,
                 m2=0.9,
                 a_start=1,
                 tau=0.9,
                 sfgrd=0.01,
                 m_inf=-np.inf,
                 min_a=1e-16,
                 callback=None,
                 callback_args=(),
                 verbose=False):
        super().__init__(f=f,
                         x=x,
                         eps=eps,
                         max_iter=max_iter,
                         max_f_eval=max_f_eval,
                         m1=m1,
                         m2=m2,
                         a_start=a_start,
                         tau=tau,
                         sfgrd=sfgrd,
                         m_inf=m_inf,
                         min_a=min_a,
                         callback=callback,
                         callback_args=callback_args,
                         verbose=verbose)
        if wf not in ('fr', 'pr', 'hs', 'dy'):
            raise ValueError(f'unknown NCG formula {wf}, choose '
                             f'one of `fr`, `pr`, `hs` and `dy`')
        self.wf = wf
        if not r_start >= 0:
            raise ValueError('r_start must be >= 0')
        self.r_start = r_start

    def minimize(self):
        last_x = np.zeros_like(self.x)  # last point visited in the line search
        last_g_x = np.zeros_like(self.x)  # gradient of last_x

        self._print_header()

        self.f_x, self.g_x = self.f.function(self.x), self.f.jacobian(self.x)
        self.ng = np.linalg.norm(self.g_x)

        if self.eps < 0:
            ng0 = -self.ng  # norm of first subgradient
        else:
            ng0 = 1  # un-scaled stopping criterion

        while True:

            # output statistics
            self._print_info()

            try:
                self.callback()
            except StopIteration:
                break

            # stopping criteria
            if self.ng <= self.eps * ng0:
                self.status = 'optimal'
                break

            if self.iter >= self.max_iter or self.f_eval > self.line_search.max_f_eval:
                self.status = 'stopped'
                break

            # compute search direction
            if self.iter == 0:  # first iteration is off-line, standard gradient
                d = -self.g_x
                if self.is_verbose():
                    print('\t\t\t', end='')
            else:  # normal iterations, use appropriate formula
                if self.r_start > 0 and self.iter % self.f.ndim * self.r_start == 0:
                    # ... unless a restart is being performed
                    beta = 0
                    if self.is_verbose():
                        print('\t(res)\t\t', end='')
                else:
                    if self.wf == 'fr':  # Fletcher-Reeves
                        beta = (self.ng / np.linalg.norm(past_g_x)) ** 2
                    elif self.wf == 'pr':  # Polak-Ribiere
                        beta = self.g_x.dot(self.g_x - past_g_x) / np.linalg.norm(past_g_x) ** 2
                        beta = max(beta, 0)
                    elif self.wf == 'hs':  # Hestenes-Stiefel
                        beta = self.g_x.dot(self.g_x - past_g_x) / (self.g_x - past_g_x).dot(past_d)
                    elif self.wf == 'dy':  # Dai-Yuan
                        beta = self.ng ** 2 / (self.g_x - past_g_x).dot(past_d)
                    if self.is_verbose():
                        print('\tbeta: {: 1.4e}'.format(beta), end='')

                if beta != 0:
                    d = -self.g_x + beta * past_d
                else:
                    d = -self.g_x

            if self.is_lagrangian_dual():
                # project the direction over the active constraints
                d[np.logical_and(self.x <= 1e-12, d < 0)] = 0

                # first, compute the maximum feasible step size max_t such that:
                #
                #   0 <= lambda[i] + max_t * d[i]   for all i
                #     -lambda[i] <= max_t * d[i]
                #     -lambda[i] / d[i] <= max_t

                idx = d < 0  # negative gradient entries
                if any(idx):
                    max_t = min(self.line_search.a_start, min(-self.x[idx] / d[idx]))
                    self.line_search.a_start = max_t

            past_g_x = self.g_x  # previous gradient
            past_d = d  # previous search direction

            phi_p0 = self.g_x.dot(d)

            # compute step size
            a, last_f_x, last_x, last_g_x, self.f_eval = self.line_search.search(
                d, self.x, last_x, last_g_x, self.f_eval, self.f_x, phi_p0, self.is_verbose())

            # stopping criteria
            if a <= self.line_search.min_a:
                self.status = 'error'
                break

            if last_f_x <= self.m_inf:
                self.status = 'unbounded'
                break

            # update new point and gradient
            self.x, self.f_x, self.g_x = last_x, last_f_x, last_g_x
            self.ng = np.linalg.norm(self.g_x)

            self.iter += 1

        if self.verbose:
            print('\n')

        if self.is_lagrangian_dual():
            assert all(self.x >= 0)  # Lagrange multipliers

        return self
