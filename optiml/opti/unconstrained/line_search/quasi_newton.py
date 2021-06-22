import warnings

import numpy as np

from . import LineSearchOptimizer


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

    def __init__(self,
                 f,
                 x=None,
                 eps=1e-6,
                 tol=1e-8,
                 max_iter=1000,
                 max_f_eval=1000,
                 m1=0.01,
                 m2=0.9,
                 a_start=1,
                 delta=1,
                 tau=0.9,
                 sfgrd=0.01,
                 m_inf=-np.inf,
                 min_a=1e-16,
                 callback=None,
                 callback_args=(),
                 random_state=None,
                 verbose=False):
        super(BFGS, self).__init__(f=f,
                                   x=x,
                                   eps=eps,
                                   tol=tol,
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
                                   random_state=random_state,
                                   verbose=verbose)
        if not delta > 0:
            raise ValueError('delta must be > 0')
        self.delta = delta

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

            # compute approximation to Newton's direction
            if self.iter == 0:
                if self.delta > 0:
                    # initial approximation of inverse of Hessian = scaled identity
                    self.H_x = self.delta * np.eye(self.f.ndim)
                else:
                    # initial approximation of inverse of Hessian computed by finite differences of gradient
                    small_step = max(-self.delta, 1e-8)
                    self.H_x = np.zeros((self.f.ndim, self.f.ndim))
                    for i in range(self.f.ndim):
                        xp = self.x
                        xp[i] = xp[i] + small_step
                        gp = self.f.jacobian(xp)
                        self.H_x[i] = ((gp - self.g_x) / small_step).T
                    self.H_x = (self.H_x + self.H_x.T) / 2  # ensure it is symmetric
                    lambda_n = np.linalg.eigvalsh(self.H_x)[0]  # smallest eigenvalue
                    if lambda_n < 1e-6:
                        self.H_x = self.H_x + (1e-6 - lambda_n) * np.eye(self.f.ndim)
                    self.H_x = np.linalg.inv(self.H_x)

            # compute search direction
            d = -self.H_x.dot(self.g_x)

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

            # update approximation of the Hessian using the BFGS formula
            s = (last_x - self.x).reshape(self.f.ndim, 1)  # s^i = x^{i + 1} - x^i
            y = (last_g_x - self.g_x).reshape(self.f.ndim, 1)  # y^i = \nabla f(x^{i + 1}) - \nabla f(x^i)

            rho = y.T.dot(s).item()
            if rho < 1e-16:
                warnings.warn('error: y^i s^i = {: 1.4e}'.format(rho))
                self.status = 'error'
                break

            rho = 1 / rho

            if self.is_verbose():
                print('\trho: {: 1.4e}'.format(rho), end='')

            D = self.H_x.dot(y) * s.T
            self.H_x += rho * ((1 + rho * y.T.dot(self.H_x).dot(y)) * (s.dot(s.T)) - D - D.T)

            # update new point and gradient
            self.x, self.f_x, self.g_x = last_x, last_f_x, last_g_x

            try:
                self.check_lagrangian_dual_optimality()
            except StopIteration:
                break

            self.ng = np.linalg.norm(self.g_x)

            self.iter += 1

        self.check_lagrangian_dual_conditions()

        if self.verbose:
            print('\n')

        return self


class LBFGS(BFGS):

    def __init__(self,
                 f,
                 x=None,
                 eps=1e-6,
                 max_iter=1000,
                 max_f_eval=1000,
                 m1=0.01,
                 m2=0.9,
                 a_start=1,
                 delta=1,
                 tau=0.9,
                 sfgrd=0.01,
                 m_inf=-np.inf,
                 min_a=1e-16,
                 callback=None,
                 callback_args=(),
                 verbose=False):
        super(LBFGS, self).__init__(f=f,
                                    x=x,
                                    eps=eps,
                                    max_iter=max_iter,
                                    max_f_eval=max_f_eval,
                                    m1=m1,
                                    m2=m2,
                                    a_start=a_start,
                                    delta=delta,
                                    tau=tau,
                                    sfgrd=sfgrd,
                                    m_inf=m_inf,
                                    min_a=min_a,
                                    callback=callback,
                                    callback_args=callback_args,
                                    verbose=verbose)

    def minimize(self):
        raise NotImplementedError
