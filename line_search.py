def armijo_wolfe_line_search(f, d, x, last_x, last_d, f_eval, max_f_eval, min_a, sfgrd,
                             phi0=None, phi_p0=None, _as=None, m1=None, m2=None, tau=None, verbose=True):
    # Performs an Armijo-Wolfe Line Search.
    #
    # phi0 = phi(0), phi_p0 = phi'(0) < 0
    #
    # as > 0 is the first value to be tested: if phi'(as) < 0 then as is
    # divided by tau < 1 (hence it is increased) until this does not happen
    # any longer
    #
    # m1 and m2 are the standard Armijo-Wolfe parameters; note that the strong
    # Wolfe condition is used
    #
    # returns the optimal step and the optimal f-value
    ls_iter = 1  # count iterations of first phase
    while f_eval <= max_f_eval:
        phi_a, phi_ps, last_x, last_d, f_eval = f2phi(f, d, x, f_eval, _as)
        # Armijo and strong Wolfe conditions
        if (phi_a <= phi0 + m1 * _as * phi_p0) and (abs(phi_ps) <= -m2 * phi_p0):
            if verbose:
                print('\t{:2d}'.format(ls_iter), end='')
            break

        if phi_ps >= 0:
            break

        _as = _as / tau
        ls_iter += 1

    if verbose:
        print('\t{:2d}'.format(ls_iter), end='')
    ls_iter = 1  # count iterations of second phase

    am = 0
    a = _as
    phi_pm = phi_p0
    while f_eval <= max_f_eval and _as - am > min_a and phi_ps > 1e-12:
        # compute the new value by safeguarded quadratic interpolation
        a = (am * phi_ps - _as * phi_pm) / (phi_ps - phi_pm)
        a = max([am + (_as - am) * sfgrd, min([_as - (_as - am) * sfgrd, a])])

        # compute phi(a)
        phi_a, phi_p, last_x, last_d, f_eval = f2phi(f, d, x, f_eval, a)
        # Armijo and strong Wolfe conditions
        if phi_a <= phi0 + m1 * a * phi_p0 and abs(phi_p) <= -m2 * phi_p0:
            if verbose:
                print('{:2d}'.format(ls_iter), end='')
            return a, phi_a, last_x, last_d, f_eval

        # restrict the interval based on sign of the derivative in a
        if phi_p < 0:
            am = a
            phi_pm = phi_p
        else:
            _as = a
            if _as <= min_a:
                if verbose:
                    print('{:2d}'.format(ls_iter), end='')
                return a, phi_a, last_x, last_d, f_eval

            phi_ps = phi_p

        ls_iter += 1

    if verbose:
        print('{:2d}'.format(ls_iter), end='')

    return a, phi_a, last_x, last_d, f_eval


def backtracking_line_search(f, d, x, last_x, last_d, f_eval, max_f_eval, min_a,
                             phi0=None, phi_p0=None, _as=None, m1=None, tau=None, verbose=True):
    """
    Performs a Backtracking Line Search.

        phi0 = phi(0), phi_p0 = phi'(0) < 0

    as > 0 is the first value to be tested, which is decreased by multiplying
    it by tau < 1 until the Armijo condition with parameter m1 is satisfied
    :returns: the optimal step and the optimal f-value
    """

    ls_iter = 1  # count ls iterations
    while f_eval <= max_f_eval and _as > min_a:
        phi_a, _, last_x, last_d, f_eval = f2phi(f, d, x, f_eval, _as)
        if phi_a <= phi0 + m1 * _as * phi_p0:  # Armijo condition
            return _as, phi_a, last_x, last_d, f_eval

        _as = _as * tau
        ls_iter += 1

    if verbose:
        print('\t{:2d}'.format(ls_iter))

    return _as, phi_a, last_x, last_d, f_eval


def f2phi(f, d, x, f_eval, alpha=None):
    # phi(alpha) = f(x - alpha * d)
    # phi'(alpha) = < \nabla f(x - alpha * d) , d >

    last_x = x - alpha * d
    phi, last_d = f.function(last_x), f.derivative(last_x)
    phi_p = -d.T.dot(last_d).item()
    f_eval += 1
    return phi, phi_p, last_x, last_d, f_eval
