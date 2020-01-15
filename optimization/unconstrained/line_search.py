import numpy as np


class LineSearch:

    def __init__(self, f, max_f_eval=1000, m1=0.01, a_start=1, tau=0.9, min_a=1e-16, verbose=False):
        """

        :param f:          the objective function.
        :param max_f_eval: (integer scalar, optional, default value 1000): the maximum number of
                           function evaluations (hence, iterations will be not more than max_f_eval
                           because at each iteration at least a function evaluation is performed,
                           possibly more due to the line search).
        :param m1:         (real scalar, optional, default value 0.01): first parameter of the
                           Armijo-Wolfe-type line search (sufficient decrease). Has to be in (0,1).
        :param a_start:    (real scalar, optional, default value 1): starting value of alpha in the
                           line search (> 0).
        :param tau:        (real scalar, optional, default value 0.9): scaling parameter for the line
                           search. In the Armijo-Wolfe line search it is used in the first phase: if the
                           derivative is not positive, then the step is divided by tau (which is < 1,
                           hence it is increased). In the Backtracking line search, each time the step is
                           multiplied by tau (hence it is decreased).
        :param min_a:      (real scalar, optional, default value 1e-16): if the algorithm determines a step size
                           value <= min_a, this is taken as an indication that something has gone wrong (the gradient
                           is not a direction of descent, so maybe the function is not differentiable) and computation
                           is stopped. It is legal to take min_a = 0, thereby in fact skipping this test.
        :param verbose:    (boolean, optional, default value False): print details about each iteration
                           if True, nothing otherwise.
        """
        self.f = f
        if not np.isscalar(max_f_eval):
            raise ValueError('max_f_eval is not an integer scalar')
        self.max_f_eval = max_f_eval
        if not np.isscalar(m1):
            raise ValueError('m1 is not a real scalar')
        if m1 <= 0 or m1 >= 1:
            raise ValueError('m1 is not in (0,1)')
        self.m1 = m1
        if not np.isscalar(a_start):
            raise ValueError('a_start is not a real scalar')
        if a_start < 0:
            raise ValueError('a_start must be > 0')
        self.a_start = a_start
        if not np.isscalar(tau):
            raise ValueError('tau is not a real scalar')
        if tau <= 0 or tau >= 1:
            raise ValueError('tau is not in (0,1)')
        self.tau = tau
        if not np.isscalar(min_a):
            raise ValueError('min_a is not a real scalar')
        if min_a < 0:
            raise ValueError('min_a is < 0')
        self.min_a = min_a
        self.verbose = verbose

    def search(self, d, wrt, last_wrt, last_g, f_eval, phi0=None, phi_p0=None):
        return NotImplementedError


class ArmijoWolfe(LineSearch):
    """
    Performs an Armijo-Wolfe Line Search.

        phi0 = phi(0), phi_p0 = phi'(0) < 0

    a_start > 0 is the first value to be tested: if phi'(as) < 0 then
    a_start is divided by tau < 1 (hence it is increased) until this
    does not happen any longer.
    m1 and m2 are the standard Armijo-Wolfe parameters;
    note that the strong Wolfe condition is used.
    :returns: the optimal step and the optimal f-value
    """

    def __init__(self, f, max_f_eval=1000, m1=0.01, m2=0.9, a_start=1, tau=0.9, sfgrd=0.01, min_a=1e-16, verbose=False):
        """

        :param f:          the objective function.
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
                           [as * (1 + sfgrd) , am * (1 - sfgrd)], being [as , am] the current interval, whatever
                           quadratic interpolation says. If you experience problems with the line search taking
                           too many iterations to converge at "nasty" points, try to increase this.
        :param min_a:      (real scalar, optional, default value 1e-16): if the algorithm determines a step size
                           value <= min_a, this is taken as an indication that something has gone wrong (the gradient
                           is not a direction of descent, so maybe the function is not differentiable) and computation
                           is stopped. It is legal to take min_a = 0, thereby in fact skipping this test.
        :param verbose:    (boolean, optional, default value False): print details about each iteration
                           if True, nothing otherwise.
        """
        super().__init__(f, max_f_eval, m1, a_start, tau, min_a, verbose)
        if not np.isscalar(sfgrd):
            raise ValueError('sfgrd is not a real scalar')
        if sfgrd <= 0 or sfgrd >= 1:
            raise ValueError('sfgrd is not in (0,1)')
        self.sfgrd = sfgrd
        if not np.isscalar(m2):
            raise ValueError('m2 is not a real scalar')
        self.m2 = m2

    def search(self, d, wrt, last_wrt, last_g, f_eval, phi0=None, phi_p0=None):

        def f2phi(f, d, x, a, f_eval):
            # phi(a) = f(x + a * d)
            # phi'(a) = <\nabla f(x + a * d), d>

            last_wrt = x + a * d
            phi_a, last_g = f.function(last_wrt), f.jacobian(last_wrt)
            phi_p = d.T.dot(last_g)
            f_eval += 1
            return phi_a, phi_p, last_wrt, last_g, f_eval

        _as = self.a_start
        ls_iter = 1  # count iterations of first phase
        while f_eval <= self.max_f_eval:
            phi_a, phi_ps, last_wrt, last_g, f_eval = f2phi(self.f, d, wrt, _as, f_eval)
            # Armijo and strong Wolfe conditions
            if phi_a <= phi0 + self.m1 * _as * phi_p0 and abs(phi_ps) <= -self.m2 * phi_p0:
                if self.verbose:
                    print('\t{:2d}\t{:2d}'.format(ls_iter, 0), end='')
                return _as, phi_a, last_wrt, last_g, f_eval

            if phi_ps >= 0:
                break

            _as /= self.tau
            ls_iter += 1

        if self.verbose:
            print('\t{:2d}\t'.format(ls_iter), end='')
        ls_iter = 1  # count iterations of second phase

        am = 0
        a = _as
        phi_pm = phi_p0
        while f_eval <= self.max_f_eval and _as - am > self.min_a and phi_ps > 1e-12:
            # compute the new value by safeguarded quadratic interpolation
            a = (am * phi_ps - _as * phi_pm) / (phi_ps - phi_pm)

            # a = max(am * (1 + self.sfgrd), min(_as * (1 - self.sfgrd), a))
            a = max(am + (_as - am) * self.sfgrd, min(_as - (_as - am) * self.sfgrd, a))

            # compute phi(a)
            phi_a, phi_p, last_wrt, last_g, f_eval = f2phi(self.f, d, wrt, a, f_eval)
            # Armijo and strong Wolfe conditions
            if phi_a <= phi0 + self.m1 * a * phi_p0 and abs(phi_p) <= -self.m2 * phi_p0:
                break

            # restrict the interval based on sign of the derivative in a
            if phi_p < 0:
                am = a
                phi_pm = phi_p
            else:
                _as = a
                if _as <= self.min_a:
                    break

                phi_ps = phi_p

            ls_iter += 1

        if self.verbose:
            print('{:2d}'.format(ls_iter), end='')
        return a, phi_a, last_wrt, last_g, f_eval


class Backtracking(LineSearch):
    """
    Performs a Backtracking Line Search.

        phi0 = phi(0), phi_p0 = phi'(0) < 0

    a_start > 0 is the first value to be tested, which is decreased by multiplying
    it by tau < 1 until the Armijo condition with parameter m1 is satisfied.
    :returns: the optimal step and the optimal f-value
    """

    def __init__(self, f, max_f_eval=1000, m1=0.01, a_start=1, tau=0.9, min_a=1e-16, verbose=False):
        """

        :param f:          the objective function.
        :param max_f_eval: (integer scalar, optional, default value 1000): the maximum number of
                           function evaluations (hence, iterations will be not more than max_f_eval
                           because at each iteration at least a function evaluation is performed,
                           possibly more due to the line search).
        :param m1:         (real scalar, optional, default value 0.01): first parameter of the
                           Armijo-Wolfe-type line search (sufficient decrease). Has to be in (0,1).
        :param a_start:    (real scalar, optional, default value 1): starting value of alpha in the
                           line search (> 0).
        :param tau:        (real scalar, optional, default value 0.9): scaling parameter for the line
                           search. In the Armijo-Wolfe line search it is used in the first phase: if the
                           derivative is not positive, then the step is divided by tau (which is < 1,
                           hence it is increased). In the Backtracking line search, each time the step is
                           multiplied by tau (hence it is decreased).
        :param min_a:      (real scalar, optional, default value 1e-16): if the algorithm determines a step size
                           value <= min_a, this is taken as an indication that something has gone wrong (the gradient
                           is not a direction of descent, so maybe the function is not differentiable) and computation
                           is stopped. It is legal to take min_a = 0, thereby in fact skipping this test.
        :param verbose:    (boolean, optional, default value False): print details about each iteration
                           if True, nothing otherwise.
        """
        super().__init__(f, max_f_eval, m1, a_start, tau, min_a, verbose)

    def search(self, d, wrt, last_wrt, last_g, f_eval, phi0=None, phi_p0=None):

        def f2phi(f, d, x, a, f_eval):
            # phi(a) = f(x + a * d)

            last_wrt = x + a * d
            phi_a, last_g = f.function(last_wrt), f.jacobian(last_wrt)
            f_eval += 1
            return phi_a, last_wrt, last_g, f_eval

        _as = self.a_start
        ls_iter = 1  # count ls iterations
        while f_eval <= self.max_f_eval and _as > self.min_a:
            phi_a, last_wrt, last_g, f_eval = f2phi(self.f, d, wrt, _as, f_eval)
            if phi_a <= phi0 + self.m1 * _as * phi_p0:  # Armijo condition
                break

            _as *= self.tau
            ls_iter += 1

        if self.verbose:
            print('\t{:2d}'.format(ls_iter), end='')
        return _as, phi_a, last_wrt, last_g, f_eval
