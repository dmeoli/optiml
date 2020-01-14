import numpy as np


class LineSearchWarning(RuntimeWarning):
    pass


def dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """

     Subroutine dcstep

     This subroutine computes a safeguarded step for a search
     procedure and updates an interval that contains a step that
     satisfies a sufficient decrease and a curvature condition.

     The parameter stx contains the step with the least function
     value. If brackt is set to True: a minimizer has
     been bracketed in an interval with endpoints stx and sty.
     The parameter stp contains the current step.
     The subroutine assumes that if brackt is set to True:

           min(stx,sty) < stp < max(stx,sty),

     and that the derivative at stx is negative in the direction
     of the step.

     The subroutine statement is

       subroutine dcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,
                         stpmin,stpmax)

     where

       stx is a double precision variable.
         On entry stx is the best step obtained so far and is an
            endpoint of the interval that contains the minimizer.
         On exit stx is the updated best step.

       fx is a double precision variable.
         On entry fx is the function at stx.
         On exit fx is the function at stx.

       dx is a double precision variable.
         On entry dx is the derivative of the function at
            stx. The derivative must be negative in the direction of
            the step, that is, dx and stp - stx must have opposite
            signs.
         On exit dx is the derivative of the function at stx.

       sty is a double precision variable.
         On entry sty is the second endpoint of the interval that
            contains the minimizer.
         On exit sty is the updated endpoint of the interval that
            contains the minimizer.

       fy is a double precision variable.
         On entry fy is the function at sty.
         On exit fy is the function at sty.

       dy is a double precision variable.
         On entry dy is the derivative of the function at sty.
         On exit dy is the derivative of the function at the exit sty.

       stp is a double precision variable.
         On entry stp is the current step. If brackt is set to True
           : on input stp must be between stx and sty.
         On exit stp is a new trial step.

       fp is a double precision variable.
         On entry fp is the function at stp
         On exit fp is unchanged.

       dp is a double precision variable.
         On entry dp is the the derivative of the function at stp.
         On exit dp is unchanged.

       brackt is an logical variable.
         On entry brackt specifies if a minimizer has been bracketed.
            Initially brackt must be set to .false.
         On exit brackt specifies if a minimizer has been bracketed.
            When a minimizer is bracketed brackt is set to True

       stpmin is a double precision variable.
         On entry stpmin is a lower bound for the step.
         On exit stpmin is unchanged.

       stpmax is a double precision variable.
         On entry stpmax is an upper bound for the step.
         On exit stpmax is unchanged.

     MINPACK-1 Project. June 1983
     Argonne National Laboratory.
     Jorge J. More' and David J. Thuente.

     MINPACK-2 Project. November 1993.
     Argonne National Laboratory and University of Minnesota.
     Brett M. Averick and Jorge J. More'.
    """

    sgnd = dp * (dx / abs(dx))

    # First case: A higher function value. The minimum is bracketed.
    # If the cubic step is closer to stx than the quadratic step, the
    # cubic step is taken, otherwise the average of the cubic and
    # quadratic steps is taken.
    if fp > fx:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        if abs(stpc - stx) < abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True

    # Second case: A lower function value and derivatives of opposite
    # sign. The minimum is bracketed. If the cubic step is farther from
    # stp than the secant step, the cubic step is taken, otherwise the
    # secant step is taken.
    elif sgnd < 0:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))

        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True

    # Third case: A lower function value, derivatives of the same sign,
    # and the magnitude of the derivative decreases.
    elif abs(dp) < abs(dx):
        # The cubic step is computed only if the cubic tends to infinity
        # in the direction of the step or if the minimum of the cubic
        # is beyond stp. Otherwise the cubic step is defined to be the
        # secant step.
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))

        # The case gamma = 0 only arises if the cubic does not tend
        # to infinity in the direction of the step.
        gamma = s * np.sqrt(max(0, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if r < 0 and gamma != 0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin

        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if brackt:
            # A minimizer has been bracketed. If the cubic step is
            # closer to stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq

            if stp > stx:
                stpf = min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = max(stp + 0.66 * (sty - stp), stpf)
        else:
            # A minimizer has not been bracketed. If the cubic step is
            # farther from stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.

            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = min(stpmax, stpf)
            stpf = max(stpmin, stpf)

    # Fourth case: A lower function value, derivatives of the same sign,
    # and the magnitude of the derivative does not decrease. If the
    # minimum is not bracketed, the step is either stpmin or stpmax,
    # otherwise the cubic step is taken.
    else:
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))

            gamma = s * np.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc

        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # Update the interval which contains a minimizer.
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0:
            sty = stx
            fy = fx
            dy = dx

        stx = stp
        fx = fp
        dx = dp

    # Compute the new step.
    stp = stpf

    return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax


def dcsrch(stp, f, g, ftol, gtol, xtol, task, stpmin, stpmax, isave, dsave):
    """
     Subroutine dcsrch

     This subroutine finds a step that satisfies a sufficient
     decrease condition and a curvature condition.

     Each call of the subroutine updates an interval with
     endpoints stx and sty. The interval is initially chosen
     so that it contains a minimizer of the modified function

           psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).

     If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
     interval is chosen so that it contains a minimizer of f.

     The algorithm is designed to find a step that satisfies
     the sufficient decrease condition

           f(stp) <= f(0) + ftol*stp*f'(0),

     and the curvature condition

           abs(f'(stp)) <= gtol*abs(f'(0)).

     If ftol is less than gtol and if, for example, the function
     is bounded below, then there is always a step which satisfies
     both conditions.

     If no step can be found that satisfies both conditions, then
     the algorithm stops with a warning. In this case stp only
     satisfies the sufficient decrease condition.

     A typical invocation of dcsrch has the following outline:

     Evaluate the function at stp = 0.0d0; store in f.
     Evaluate the gradient at stp = 0.0d0; store in g.
     Choose a starting step stp.

     task = 'START'
  10 continue
        call dcsrch(stp,f,g,ftol,gtol,xtol,task,stpmin,stpmax,
    +               isave,dsave)
        if (task == 'FG') then
           Evaluate the function and the gradient at stp
           go to 10
           end if

     NOTE: The user must not alter work arrays between calls.

     The subroutine statement is

       subroutine dcsrch(f,g,stp,ftol,gtol,xtol,stpmin,stpmax,
                         task,isave,dsave)
     where

       stp is a double precision variable.
         On entry stp is the current estimate of a satisfactory
            step. On initial entry, a positive initial estimate
            must be provided.
         On exit stp is the current estimate of a satisfactory step
            if task = 'FG'. If task = 'CONV' then stp satisfies
            the sufficient decrease and curvature condition.

       f is a double precision variable.
         On initial entry f is the value of the function at 0.
            On subsequent entries f is the value of the
            function at stp.
         On exit f is the value of the function at stp.

       g is a double precision variable.
         On initial entry g is the derivative of the function at 0.
            On subsequent entries g is the derivative of the
            function at stp.
         On exit g is the derivative of the function at stp.

       ftol is a double precision variable.
         On entry ftol specifies a nonnegative tolerance for the
            sufficient decrease condition.
         On exit ftol is unchanged.

       gtol is a double precision variable.
         On entry gtol specifies a nonnegative tolerance for the
            curvature condition.
         On exit gtol is unchanged.

       xtol is a double precision variable.
         On entry xtol specifies a nonnegative relative tolerance
            for an acceptable step. The subroutine exits with a
            warning if the relative difference between sty and stx
            is less than xtol.
         On exit xtol is unchanged.

       task is a character variable of length at least 60.
         On initial entry task must be set to 'START'.
         On exit task indicates the required action:

            If task(1:2) = 'FG' then evaluate the function and
            derivative at stp and call dcsrch again.

            If task(1:4) = 'CONV' then the search is successful.

            If task(1:4) = 'WARN' then the subroutine is not able
            to satisfy the convergence conditions. The exit value of
            stp contains the best point found during the search.

            If task(1:5) = 'ERROR' then there is an error in the
            input arguments.

         On exit with convergence, a warning or an error, the
            variable task contains additional information.

       stpmin is a double precision variable.
         On entry stpmin is a nonnegative lower bound for the step.
         On exit stpmin is unchanged.

       stpmax is a double precision variable.
         On entry stpmax is a nonnegative upper bound for the step.
         On exit stpmax is unchanged.

       isave is an integer work array of dimension 2.

       dsave is a double precision work array of dimension 13.

     Subprograms called

       MINPACK-2 ... dcstep

     MINPACK-1 Project. June 1983.
     Argonne National Laboratory.
     Jorge J. More' and David J. Thuente.

     MINPACK-2 Project. November 1993.
     Argonne National Laboratory and University of Minnesota.
     Brett M. Averick, Richard G. Carter, and Jorge J. More'.
    """

    xtrapl = 1.1
    xtrapu = 4.0

    # Initialization block.
    if task == 'START':

        if stpmin is None:
            stpmin = 0
        if stpmax is None:
            stpmax = 1e999

        # Check the input arguments for errors.
        if stp < stpmin:
            task = 'Error: stp < stpmin'
        if stp > stpmax:
            task = 'Error: stp > stpmax'
        if g >= 0.0:
            task = 'Error: initial %e >= 0' % g
        if ftol < 0.0:
            task = 'Error: ftol < zero'
        if gtol < 0.0:
            task = 'Error: gtol < zero'
        if xtol < 0.0:
            task = 'Error: xtol < zero'
        if stpmin < 0.0:
            task = 'Error: stpmin < zero'
        if stpmax < stpmin:
            task = 'Error: stpmax < stpmin'

        # Exit if there are errors on input.
        if 'Error' in task:
            return stp, f, g, task

        # Initialize local variables.
        brackt = False
        stage = 1
        finit = f
        ginit = g
        gtest = ftol * ginit
        width = stpmax - stpmin
        width1 = width / 0.5

        # The variables stx, fx, gx contain the values of the step,
        # function, and derivative at the best step.
        # The variables sty, fy, gy contain the value of the step,
        # function, and derivative at sty.
        # The variables stp, f, g contain the values of the step,
        # function, and derivative at stp.

        stx = 0.0
        fx = finit
        gx = ginit
        sty = 0.0
        fy = finit
        gy = ginit
        stmin = 0.0
        stmax = stp + xtrapu * stp
        task = 'FG'

        # Save local variables.
        isave[0] = brackt
        isave[1] = stage
        dsave[0] = ginit
        dsave[1] = gtest
        dsave[2] = gx
        dsave[3] = gy
        dsave[4] = finit
        dsave[5] = fx
        dsave[6] = fy
        dsave[7] = stx
        dsave[8] = sty
        dsave[9] = stmin
        dsave[10] = stmax
        dsave[11] = width
        dsave[12] = width1

        return stp, f, g, task

    else:

        # Restore local variables.
        brackt = isave[1]
        stage = isave[1]
        ginit = dsave[0]
        gtest = dsave[1]
        gx = dsave[2]
        gy = dsave[3]
        finit = dsave[4]
        fx = dsave[5]
        fy = dsave[6]
        stx = dsave[7]
        sty = dsave[8]
        stmin = dsave[9]
        stmax = dsave[10]
        width = dsave[11]
        width1 = dsave[12]

    # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
    # algorithm enters the second stage.
    ftest = finit + stp * gtest
    if stage == 1 and f <= ftest and g >= 0.0:
        stage = 2

    # Test for warnings.
    if brackt and (stp <= stmin or stp >= stmax):
        task = 'Warning: Rounding errors prevent progress'
    if brackt and stmax - stmin <= xtol * stmax:
        task = 'Warning: xtol test satisfied'
    if stp == stpmax and f <= ftest and g <= gtest:
        task = 'Warning: stp = stpmax'
    if stp == stpmin and (f > ftest or g >= gtest):
        task = 'Warning: stp = stpmin'

    # Test for convergence.
    if f <= ftest and abs(g) <= gtol * (-ginit):
        task = 'Convergence'

    # Test for termination.
    if 'Warning' in task or task == 'Convergence':
        # Save local variables.
        isave[0] = brackt
        isave[1] = stage
        dsave[0] = ginit
        dsave[1] = gtest
        dsave[2] = gx
        dsave[3] = gy
        dsave[4] = finit
        dsave[5] = fx
        dsave[6] = fy
        dsave[7] = stx
        dsave[8] = sty
        dsave[9] = stmin
        dsave[10] = stmax
        dsave[11] = width
        dsave[12] = width1
        return stp, f, g, task

    # A modified function is used to predict the step during the
    # first stage if a lower function value has been obtained but
    # the decrease is not sufficient.
    if stage == 1 and fx >= f > ftest:
        # Define the modified function and derivative values.
        fm = f - stp * gtest
        fxm = fx - stx * gtest
        fym = fy - sty * gtest
        gm = g - gtest
        gxm = gx - gtest
        gym = gy - gtest

        # Call dcstep to update stx, sty, and to compute the new step.
        stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax = \
            dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax)

        # Reset the function and derivative values for f.
        fx = fxm + stx * gtest
        fy = fym + sty * gtest
        gx = gxm + gtest
        gy = gym + gtest

    else:
        # Call dcstep to update stx, sty, and to compute the new step.
        stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax = \
            dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax)

    # Decide if a bisection step is needed.
    if brackt:
        if abs(sty - stx) >= 0.66 * width1:
            stp = stx + 0.5 * (sty - stx)
        width1 = width
        width = abs(sty - stx)

    # Set the minimum and maximum steps allowed for stp.
    if brackt:
        stmin = min(stx, sty)
        stmax = max(stx, sty)
    else:
        stmin = stp + xtrapl * (stp - stx)
        stmax = stp + xtrapu * (stp - stx)

    # Force the step to be within the bounds stpmax and stpmin.
    stp = max(stp, stpmin)
    stp = min(stp, stpmax)

    # If further progress is not possible, let stp be the best
    # point obtained during the search.
    if brackt and (stp <= stmin or stp >= stmax) or (brackt and stmax - stmin <= xtol * stmax):
        stp = stx

    # Obtain another function and derivative.
    task = 'FG'

    # Save local variables.
    isave[0] = brackt
    isave[1] = stage
    dsave[0] = ginit
    dsave[1] = gtest
    dsave[2] = gx
    dsave[3] = gy
    dsave[4] = finit
    dsave[5] = fx
    dsave[6] = fy
    dsave[7] = stx
    dsave[8] = sty
    dsave[9] = stmin
    dsave[10] = stmax
    dsave[11] = width
    dsave[12] = width1
    return stp, f, g, task


def line_search_wolfe(f, fprime, xk, pk, gfk=None,
                      old_fval=None, old_old_fval=None,
                      args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                      xtol=1e-14):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`
    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`
    The rest of the parameters are the same as for `scalar_search_wolfe1`.
    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point
    """
    if gfk is None:
        gfk = fprime(xk)

    if isinstance(fprime, tuple):
        eps = fprime[1]
        fprime = fprime[0]
        newargs = (f, eps) + args
        gradient = False
    else:
        newargs = args
        gradient = True

    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return f(xk + s * pk, *args)

    def derphi(s):
        gval[0] = fprime(xk + s * pk, *newargs)
        if gradient:
            gc[0] += 1
        else:
            fc[0] += len(xk) + 1
        return np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe(
        phi, derphi, old_fval, old_old_fval, derphi0,
        c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]


def scalar_search_wolfe(phi, derphi, phi0=None, old_phi0=None, derphi0=None,
                        c1=1e-4, c2=0.9,
                        amax=50, amin=1e-8, xtol=1e-14):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions
    alpha > 0 is assumed to be a descent direction.
    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax, amin : float, optional
        Maximum and minimum step size
    xtol : float, optional
        Relative tolerance for an acceptable step.
    Returns
    -------
    alpha : float
        Step size, or None if no suitable step was found
    phi : float
        Value of `phi` at the new point `alpha`
    phi0 : float
        Value of `phi` at `alpha=0`
    Notes
    -----
    Uses routine DCSRCH from MINPACK.
    """

    if phi0 is None:
        phi0 = phi(0.)
    if derphi0 is None:
        derphi0 = derphi(0.)

    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    phi1 = phi0
    derphi1 = derphi0
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = 'START'

    maxiter = 100
    for i in range(maxiter):
        stp, phi1, derphi1, task = dcsrch(alpha1, phi1, derphi1,
                                          c1, c2, xtol, task,
                                          amin, amax, isave, dsave)

        if 'FG' in str(task):
            alpha1 = stp
            phi1 = phi(stp)
            derphi1 = derphi(stp)
        else:
            break
    else:
        # maxiter reached, the line search did not converge
        stp = None

    if 'ERROR' in str(task) or 'WARN' in str(task):
        stp = None  # failed

    return stp, phi1, phi0
