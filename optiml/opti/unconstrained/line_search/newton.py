import numpy as np

from . import LineSearchOptimizer


class Newton(LineSearchOptimizer):
    # Apply a classical Newton's method for the minimization of the provided
    # function f.
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
    # - delta (real scalar, optional, default value 1e-6): minimum positive
    #   value for the eigenvalues of the modified Hessian used to compute the
    #   Newton direction
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
    # - mina (real scalar, optional, default value 1e-16): if the algorithm
    #   determines a step size value <= mina, this is taken as an indication
    #   that something has gone wrong (the gradient is not a direction of
    #   descent, so maybe the function is not differentiable) and computation
    #   is stopped. It is legal to take mina = 0, thereby in fact skipping this
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
    #     continuing optimization (see mina above)

    def __init__(self,
                 f,
                 x,
                 eps=1e-6,
                 max_iter=1000,
                 max_f_eval=1000,
                 m1=0.01,
                 m2=0.9,
                 a_start=1,
                 delta=1e-6,
                 tau=0.9,
                 sfgrd=0.01,
                 m_inf=-np.inf,
                 min_a=1e-12,
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
        self.H_x = np.zeros(0)
        if not delta > 0:
            raise ValueError('delta must be > 0')
        self.delta = delta

    def minimize(self):
        last_x = np.zeros(self.f.ndim)  # last point visited in the line search
        last_g = np.zeros(self.f.ndim)  # gradient of last_x

        if self.verbose:
            print('iter\tfeval\t cost\t\t gnorm', end='')
            if self.f.f_star() < np.inf:
                print('\t\t gap\t\t rate', end='')
                prev_v = np.inf

        while True:
            self.f_x, self.g_x, self.H_x = self.f.function(self.x), self.f.jacobian(self.x), self.f.hessian(self.x)
            ng = np.linalg.norm(self.g_x)

            if self.eps < 0:
                ng0 = -ng  # norm of first subgradient
            else:
                ng0 = 1  # un-scaled stopping criterion

            if self.is_verbose():
                print('\n{:4d}\t{:4d}\t{: 1.4e}\t{: 1.4e}'.format(self.iter, self.f_eval, self.f_x, ng), end='')
                if self.f.f_star() < np.inf:
                    print('\t{: 1.4e}'.format(self.f_x - self.f.f_star()), end='')
                    if prev_v < np.inf:
                        print('\t{: 1.4e}'.format((self.f_x - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    else:
                        print('\t\t', end='')
                    prev_v = self.f_x

            try:
                self.callback()
            except StopIteration:
                break

            # stopping criteria
            if ng <= self.eps * ng0:
                self.status = 'optimal'
                break

            if self.iter > self.max_iter or self.f_eval > self.line_search.max_f_eval:
                self.status = 'stopped'
                break

            # compute Newton's direction
            lambda_n = min(np.linalg.eigvalsh(self.H_x))  # smallest eigenvalue
            if lambda_n < self.delta:
                if self.is_verbose():
                    print('\tdelta: {: 1.4e}'.format(self.delta - lambda_n), end='')
                self.H_x = self.H_x + (self.delta - lambda_n) * np.identity(len(self.g_x))
            else:
                if self.is_verbose():
                    print('\tdelta: {: 1.4e}'.format(self.delta), end='')

            d = -np.linalg.inv(self.H_x).dot(self.g_x)

            phi_p0 = self.g_x.T.dot(d)

            # compute step size: in Newton's method, the default initial step size is 1
            a, self.f_x, last_x, last_g, self.f_eval = self.line_search.search(
                d, self.x, last_x, last_g, self.f_eval, self.f_x, phi_p0, self.is_verbose())

            # output statistics
            if self.is_verbose():
                print('\tastar: {: 1.4e}'.format(a), end='')

            if a <= self.line_search.min_a:
                self.status = 'error'
                break

            if self.f_x <= self.m_inf:
                self.status = 'unbounded'
                break

            # update new point
            self.x = last_x

            self.iter += 1

        if self.verbose:
            print('\n')

        return self
