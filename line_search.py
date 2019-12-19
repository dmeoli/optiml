def armijo_wolfe_line_search(f, d, x, last_x, last_d, f_eval, max_f_eval, min_a, sfgrd, phi0=None,
                             phi_p0=None, a_start=None, m1=None, m2=None, tau=None, verbose=True):
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

    ls_iter = 1  # count iterations of first phase
    while f_eval <= max_f_eval:
        phi_a, phi_ps, last_x, last_d, f_eval = f2phi(f, d, x, f_eval, a_start)
        # Armijo and strong Wolfe conditions
        if phi_a <= phi0 + m1 * a_start * phi_p0 and abs(phi_ps) <= -m2 * phi_p0:
            if verbose:
                print('\t{:2d}\t{:2d}'.format(ls_iter, 0), end='')
            return a_start, phi_a, last_x, last_d, f_eval

        if phi_ps >= 0:
            break

        a_start /= tau
        ls_iter += 1

    if verbose:
        print('\t{:2d}\t'.format(ls_iter), end='')
    ls_iter = 1  # count iterations of second phase

    am = 0
    a = a_start
    phi_pm = phi_p0
    while f_eval <= max_f_eval and a_start - am > min_a and phi_ps > 1e-12:
        # compute the new value by safeguarded quadratic interpolation
        a = (am * phi_ps - a_start * phi_pm) / (phi_ps - phi_pm)
        a = max([am * (1 + sfgrd), min([a_start * (1 - sfgrd), a])])

        # compute phi(a)
        phi_a, phi_p, last_x, last_d, f_eval = f2phi(f, d, x, f_eval, a)
        # Armijo and strong Wolfe conditions
        if phi_a <= phi0 + m1 * a * phi_p0 and abs(phi_p) <= -m2 * phi_p0:
            break

        # restrict the interval based on sign of the derivative in a
        if phi_p < 0:
            am = a
            phi_pm = phi_p
        else:
            a_start = a
            if a_start <= min_a:
                break

            phi_ps = phi_p

        ls_iter += 1

    if verbose:
        print('{:2d}'.format(ls_iter), end='')
    return a, phi_a, last_x, last_d, f_eval


def backtracking_line_search(f, d, x, last_x, last_d, f_eval, max_f_eval, min_a, phi0=None,
                             phi_p0=None, a_start=None, m1=None, tau=None, verbose=True):
    """
    Performs a Backtracking Line Search.

        phi0 = phi(0), phi_p0 = phi'(0) < 0

    a_start > 0 is the first value to be tested, which is decreased by multiplying
    it by tau < 1 until the Armijo condition with parameter m1 is satisfied.
    :returns: the optimal step and the optimal f-value
    """

    ls_iter = 1  # count ls iterations
    while f_eval <= max_f_eval and a_start > min_a:
        phi_a, _, last_x, last_d, f_eval = f2phi(f, d, x, f_eval, a_start)
        if phi_a <= phi0 + m1 * a_start * phi_p0:  # Armijo condition
            break

        a_start *= tau
        ls_iter += 1

    if verbose:
        print('\t{:2d}'.format(ls_iter), end='')
    return a_start, phi_a, last_x, last_d, f_eval


def f2phi(f, d, x, f_eval, a):
    # phi(a) = f(x + a * d)
    # phi'(a) = <\nabla f(x + a * d) , d>

    last_x = x + a * -d
    phi_a, last_d = f.function(last_x), f.jacobian(last_x)
    phi_p = -d.T.dot(last_d).item()
    f_eval += 1
    return phi_a, phi_p, last_x, last_d, f_eval
