class LineSearch:
    def __init__(self, f, max_f_eval=1000, min_a=1e-16, a_start=1, m1=0.01, tau=0.9, verbose=False):
        self.f = f
        self.max_f_eval = max_f_eval
        self.min_a = min_a
        self.a_start = a_start
        self.m1 = m1
        self.tau = tau
        self.verbose = verbose

    def search(self, d, wrt, last_wrt, last_d, last_h, f_eval, phi0=None, phi_p0=None):
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

    def __init__(self, f, max_f_eval=1000, min_a=1e-16, sfgrd=0.01, a_start=1, m1=0.01, m2=0.9, tau=0.9, verbose=False):
        super().__init__(f, max_f_eval, min_a, a_start, m1, tau, verbose)
        self.sfgrd = sfgrd
        self.m2 = m2

    def search(self, d, wrt, last_wrt, last_d, last_h, f_eval, phi0=None, phi_p0=None):
        ls_iter = 1  # count iterations of first phase
        while f_eval <= self.max_f_eval:
            phi_a, phi_ps, last_wrt, last_d, last_h, f_eval = f2phi(self.f, d, wrt, last_h, f_eval, self.a_start)
            # Armijo and strong Wolfe conditions
            if phi_a <= phi0 + self.m1 * self.a_start * phi_p0 and abs(phi_ps) <= -self.m2 * phi_p0:
                if self.verbose:
                    print('\t{:2d}\t{:2d}'.format(ls_iter, 0), end='')
                return self.a_start, phi_a, last_wrt, last_d, last_h, f_eval

            if phi_ps >= 0:
                break

            self.a_start /= self.tau
            ls_iter += 1

        if self.verbose:
            print('\t{:2d}\t'.format(ls_iter), end='')
        ls_iter = 1  # count iterations of second phase

        am = 0
        a = self.a_start
        phi_pm = phi_p0
        while f_eval <= self.max_f_eval and self.a_start - am > self.min_a and phi_ps > 1e-12:
            # compute the new value by safeguarded quadratic interpolation
            a = max(am * (1 + self.sfgrd), min(self.a_start * (1 - self.sfgrd),
                                               (am * phi_ps - self.a_start * phi_pm) / (phi_ps - phi_pm)))

            # compute phi(a)
            phi_a, phi_p, last_wrt, last_d, last_h, f_eval = f2phi(self.f, d, wrt, last_h, f_eval, a)
            # Armijo and strong Wolfe conditions
            if phi_a <= phi0 + self.m1 * a * phi_p0 and abs(phi_p) <= -self.m2 * phi_p0:
                break

            # restrict the interval based on sign of the derivative in a
            if phi_p < 0:
                am = a
                phi_pm = phi_p
            else:
                a_start = a
                if a_start <= self.min_a:
                    break

                phi_ps = phi_p

            ls_iter += 1

        if self.verbose:
            print('{:2d}'.format(ls_iter), end='')
        return a, phi_a, last_wrt, last_d, last_h, f_eval


class Backtracking(LineSearch):
    """
    Performs a Backtracking Line Search.

        phi0 = phi(0), phi_p0 = phi'(0) < 0

    a_start > 0 is the first value to be tested, which is decreased by multiplying
    it by tau < 1 until the Armijo condition with parameter m1 is satisfied.
    :returns: the optimal step and the optimal f-value
    """

    def __init__(self, f, max_f_eval=1000, min_a=1e-16, a_start=1, m1=0.01, tau=0.9, verbose=False):
        super().__init__(f, max_f_eval, min_a, a_start, m1, tau, verbose)

    def search(self, d, wrt, last_wrt, last_d, last_h, f_eval, phi0=None, phi_p0=None):
        ls_iter = 1  # count ls iterations
        while f_eval <= self.max_f_eval and self.a_start > self.min_a:
            phi_a, _, last_wrt, last_d, last_h, f_eval = f2phi(self.f, d, wrt, last_h, f_eval, self.a_start)
            if phi_a <= phi0 + self.m1 * self.a_start * phi_p0:  # Armijo condition
                break

            self.a_start *= self.tau
            ls_iter += 1

        if self.verbose:
            print('\t{:2d}'.format(ls_iter), end='')
        return self.a_start, phi_a, last_wrt, last_d, last_h, f_eval


def f2phi(f, d, x, last_h, f_eval, a):
    # phi(a) = f(x + a * d)
    # phi'(a) = <\nabla f(x + a * d), d>

    last_wrt = x + a * d
    phi_a, last_g = f.function(last_wrt), f.jacobian(last_wrt)
    last_h = f.hessian(last_wrt) if last_h is not None else None
    phi_p = d.T.dot(last_g)
    f_eval += 1
    return phi_a, phi_p, last_wrt, last_g, last_h, f_eval
