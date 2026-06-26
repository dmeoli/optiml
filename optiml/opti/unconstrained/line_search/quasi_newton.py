import warnings

import numpy as np

from . import LineSearchOptimizer


class BFGS(LineSearchOptimizer):
    r"""
    Apply a Quasi-Newton approach, in particular using the celebrated
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) formula, for the minimization of the
    provided function f.

    Unlike Newton's method, the (inverse of the) Hessian is not computed but
    iteratively approximated by a dense [n x n] matrix H, updated at each iteration
    with the rank-two BFGS formula from the latest step :math:`s^{i} = x^{i+1} - x^{i}`
    and the latest gradient variation :math:`y^{i} = \nabla f(x^{i+1}) - \nabla f(x^{i})`,
    so that the search direction :math:`d = -H \nabla f(x)` requires no linear system
    solve; the step size is then chosen by an Armijo-Wolfe (or Backtracking) line search.
    """

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
        """

        :param f:          the objective function.
        :param x:          ([n x 1] real column vector): the point where to start the algorithm from.
        :param eps:        (real scalar, optional, default value 1e-6): the accuracy in the stopping
                           criterion: the algorithm is stopped when the norm of the gradient is less
                           than or equal to eps. If a negative value is provided, this is used in a
                           *relative* stopping criterion: the algorithm is stopped when the norm of
                           the gradient is less than or equal to (- eps) * || norm of the first gradient ||.
        :param max_f_eval: (integer scalar, optional, default value 1000): the maximum number of
                           function evaluations (hence, iterations will be not more than max_f_eval
                           because at each iteration at least a function evaluation is performed,
                           possibly more due to the line search).
        :param m1:         (real scalar, optional, default value 0.01): first parameter of the
                           Armijo-Wolfe-type line search (sufficient decrease). Has to be in (0,1).
        :param m2:         (real scalar, optional, default value 0.9): typically the second parameter
                           of the Armijo-Wolfe-type line search (strong curvature condition). It should
                           to be in (0,1); if not, it is taken to mean that the simpler Backtracking
                           line search should be used instead.
        :param a_start:    (real scalar, optional, default value 1): starting value of alpha in the
                           line search (> 0).
        :param delta:      (real scalar, optional, default value 1): the initial approximation of the
                           inverse of the Hessian is taken as delta * I if delta > 0; otherwise, the
                           initial Hessian is approximated by finite differences with - delta as the
                           step, and inverted just the once.
        :param tau:        (real scalar, optional, default value 0.9): scaling parameter for the line
                           search. In the Armijo-Wolfe line search it is used in the first phase: if the
                           derivative is not positive, then the step is divided by tau (which is < 1,
                           hence it is increased). In the Backtracking line search, each time the step is
                           multiplied by tau (hence it is decreased).
        :param sfgrd:      (real scalar, optional, default value 0.01): safeguard parameter for the line search.
                           To avoid numerical problems that can occur with the quadratic interpolation if the
                           derivative at one endpoint is too large w.r.t. The one at the other (which leads to
                           choosing a point extremely near to the other endpoint), a *safeguarded* version of
                           interpolation is used whereby the new point is chosen in the interval
                           [as * (1 + sfgrd), am * (1 - sfgrd)], being [as, am] the current interval, whatever
                           quadratic interpolation says. If you experience problems with the line search taking
                           too many iterations to converge at "nasty" points, try to increase this.
        :param m_inf:      (real scalar, optional, default value -inf): if the algorithm determines a value for
                           f() <= m_inf this is taken as an indication that the problem is unbounded below and
                           computation is stopped (a "finite -inf").
        :param min_a:      (real scalar, optional, default value 1e-16): if the algorithm determines a step size
                           value <= min_a, this is taken as an indication that something has gone wrong (the gradient
                           is not a direction of descent, so maybe the function is not differentiable) and computation
                           is stopped. It is legal to take min_a = 0, thereby in fact skipping this test.
        :param verbose:    (boolean, optional, default value False): print details about each iteration
                           if True, nothing otherwise.
        :return x:         ([n x 1] real column vector): the best solution found so far.
        :return status: (string): the status of the algorithm at termination, one of: ``optimal`` (x is a(n approximately) optimal solution); ``unbounded`` (f() was driven below m_inf, i.e., the problem looks unbounded below); ``stopped`` (the maximum number of iterations/evaluations was reached); ``error`` (a numerical error occurred, e.g., the step size fell below min_a).
        """
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
    r"""
    Apply a Limited-memory Quasi-Newton approach, in particular the Limited-memory
    Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) formula, for the minimization of the
    provided function f.

    Unlike the full BFGS, which explicitly stores and updates the [n x n] dense
    approximation of the inverse of the Hessian (hence requiring :math:`O(n^2)` memory and
    :math:`O(n^2)` operations per iteration), L-BFGS never forms that matrix: it only keeps
    the m most recent curvature pairs

    .. math::

        s^{i} = x^{i+1} - x^{i} \qquad y^{i} = \nabla f(x^{i+1}) - \nabla f(x^{i})

    dropping the oldest pair as soon as a new one is computed, and recovers the
    search direction :math:`d = -H \nabla f(x)` implicitly through the celebrated
    *two-loop recursion* (Nocedal, 1980) [1]_, which costs just :math:`O(m n)` memory and
    operations per iteration. This makes the method suited for large scale problems
    where storing a dense [n x n] matrix would be prohibitive, at the price of using
    a low-rank approximation of the inverse of the Hessian built only from the last
    m steps.

    At each iteration the recursion is restarted from a fresh initial approximation
    of the inverse of the Hessian :math:`H^{0} = \gamma I`; rather than using the fixed
    delta of the full BFGS, :math:`\gamma` is dynamically set to the scaling factor (Liu & Nocedal,
    1989) [2]_ :math:`\gamma = (s^{i-1} y^{i-1}) / (y^{i-1} y^{i-1})`, which calibrates the
    trial step to the size of the latest curvature, greatly improving the quality of
    the search direction. As long as no curvature pair has been collected yet, i.e.,
    at the very first iteration, the (scaled) steepest descent direction
    :math:`d = -\delta \nabla f(x)` is used.

    References
    ----------

    .. [1] Nocedal, J. (1980). Updating Quasi-Newton Matrices with Limited Storage.
       Mathematics of Computation, 35(151), 773-782.

    .. [2] Liu, D. C., & Nocedal, J. (1989). On the Limited Memory BFGS Method for
       Large Scale Optimization. Mathematical Programming, 45, 503-528.
    """

    def __init__(self,
                 f,
                 x=None,
                 m=20,
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
        r"""

        :param f:          the objective function.
        :param x:          ([n x 1] real column vector): the point where to start the algorithm from.
        :param m:          (integer scalar, optional, default value 20): the number of most recent
                           curvature pairs :math:`\{s^{i}, y^{i}\}` kept in memory to implicitly represent the
                           approximation of the inverse of the Hessian. Larger values yield a more
                           accurate approximation (closer to the full BFGS) at the price of more
                           memory and computation per iteration. Has to be >= 1.
        :param eps:        (real scalar, optional, default value 1e-6): the accuracy in the stopping
                           criterion: the algorithm is stopped when the norm of the gradient is less
                           than or equal to eps. If a negative value is provided, this is used in a
                           *relative* stopping criterion: the algorithm is stopped when the norm of
                           the gradient is less than or equal to (- eps) * || norm of the first gradient ||.
        :param max_f_eval: (integer scalar, optional, default value 1000): the maximum number of
                           function evaluations (hence, iterations will be not more than max_f_eval
                           because at each iteration at least a function evaluation is performed,
                           possibly more due to the line search).
        :param m1:         (real scalar, optional, default value 0.01): first parameter of the
                           Armijo-Wolfe-type line search (sufficient decrease). Has to be in (0,1).
        :param m2:         (real scalar, optional, default value 0.9): typically the second parameter
                           of the Armijo-Wolfe-type line search (strong curvature condition). It should
                           to be in (0,1); if not, it is taken to mean that the simpler Backtracking
                           line search should be used instead.
        :param a_start:    (real scalar, optional, default value 1): starting value of alpha in the
                           line search (> 0).
        :param delta:      (real scalar, optional, default value 1): scaling of the steepest descent
                           direction :math:`d = -\delta \nabla f(x)` used at the very first iteration, before
                           any curvature pair is available. Has to be > 0.
        :param tau:        (real scalar, optional, default value 0.9): scaling parameter for the line
                           search. In the Armijo-Wolfe line search it is used in the first phase: if the
                           derivative is not positive, then the step is divided by tau (which is < 1,
                           hence it is increased). In the Backtracking line search, each time the step is
                           multiplied by tau (hence it is decreased).
        :param sfgrd:      (real scalar, optional, default value 0.01): safeguard parameter for the line search.
                           To avoid numerical problems that can occur with the quadratic interpolation if the
                           derivative at one endpoint is too large w.r.t. The one at the other (which leads to
                           choosing a point extremely near to the other endpoint), a *safeguarded* version of
                           interpolation is used whereby the new point is chosen in the interval
                           [as * (1 + sfgrd), am * (1 - sfgrd)], being [as, am] the current interval, whatever
                           quadratic interpolation says. If you experience problems with the line search taking
                           too many iterations to converge at "nasty" points, try to increase this.
        :param m_inf:      (real scalar, optional, default value -inf): if the algorithm determines a value for
                           f() <= m_inf this is taken as an indication that the problem is unbounded below and
                           computation is stopped (a "finite -inf").
        :param min_a:      (real scalar, optional, default value 1e-16): if the algorithm determines a step size
                           value <= min_a, this is taken as an indication that something has gone wrong (the gradient
                           is not a direction of descent, so maybe the function is not differentiable) and computation
                           is stopped. It is legal to take min_a = 0, thereby in fact skipping this test.
        :param verbose:    (boolean, optional, default value False): print details about each iteration
                           if True, nothing otherwise.
        :return x:         ([n x 1] real column vector): the best solution found so far.
        :return status: (string): the status of the algorithm at termination, one of: ``optimal`` (x is a(n approximately) optimal solution); ``unbounded`` (f() was driven below m_inf, i.e., the problem looks unbounded below); ``stopped`` (the maximum number of iterations/evaluations was reached); ``error`` (a numerical error occurred, e.g., the step size fell below min_a).
        """
        super(LBFGS, self).__init__(f=f,
                                    x=x,
                                    eps=eps,
                                    tol=tol,
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
                                    random_state=random_state,
                                    verbose=verbose)
        if not m >= 1:
            raise ValueError('m must be >= 1')
        self.m = m

    def _two_loop_recursion(self, s, y, rho):
        # recover the search direction d = - H \nabla f(x) from the m most recent
        # curvature pairs without ever forming H, see Algorithm 7.4 in
        # Nocedal & Wright, Numerical Optimization (2nd ed.), p. 178.
        q = self.g_x.copy()
        alpha = np.zeros(len(s))
        # first loop: from the newest pair to the oldest one
        for i in reversed(range(len(s))):
            alpha[i] = rho[i] * s[i].dot(q)
            q -= alpha[i] * y[i]
        # initial approximation of the inverse of the Hessian H^0 = gamma * I
        # with the Liu & Nocedal scaling factor gamma = (s^i y^i) / (y^i y^i)
        gamma = s[-1].dot(y[-1]) / y[-1].dot(y[-1])
        r = gamma * q
        # second loop: from the oldest pair to the newest one
        for i in range(len(s)):
            beta = rho[i] * y[i].dot(r)
            r += (alpha[i] - beta) * s[i]
        return -r

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

        # the m most recent curvature pairs and the related rho^i = 1 / (y^i s^i);
        # the oldest pair is dropped as soon as a newer one is available
        s, y, rho = [], [], []

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
            if not s:
                # no curvature information yet: take a (scaled) steepest descent step
                d = -self.delta * self.g_x
            else:
                # recover the approximation to Newton's direction by the two-loop recursion
                d = self._two_loop_recursion(s, y, rho)

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

            # compute the new curvature pair
            s_i = last_x - self.x  # s^i = x^{i + 1} - x^i
            y_i = last_g_x - self.g_x  # y^i = \nabla f(x^{i + 1}) - \nabla f(x^i)

            sy = y_i.dot(s_i)
            if sy < 1e-16:
                warnings.warn('error: y^i s^i = {: 1.4e}'.format(sy))
                self.status = 'error'
                break

            # update the memory with the new curvature pair, dropping the oldest one if full
            if len(s) == self.m:
                s.pop(0)
                y.pop(0)
                rho.pop(0)
            s.append(s_i)
            y.append(y_i)
            rho.append(1 / sy)

            if self.is_verbose():
                print('\trho: {: 1.4e}'.format(1 / sy), end='')

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
