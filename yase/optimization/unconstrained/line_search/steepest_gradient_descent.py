import numpy as np

from ... import Optimizer, Quadratic
from . import LineSearchOptimizer


class QuadraticSteepestGradientDescent(Optimizer):
    """
    Apply the Steepest Gradient Descent algorithm with exact line search to the quadratic function.

        f(x) = 1/2 x^T Q x + q^T x

    :param f:        the objective function.
    :param x:        ([n x 1] real column vector): the point where to start the algorithm from
    :return x:       ([n x 1] real column vector): either the best solution found so far (possibly the
                     optimal one) or a direction proving the problem is unbounded below, depending on case
    :return status:  (string): a string describing the status of the algorithm at termination:
                        - 'optimal': the algorithm terminated having proven that x is a(n approximately) optimal
                     solution, i.e., the norm of the gradient at x is less than the required threshold;
                        - 'unbounded': the algorithm terminated having proven that the problem is unbounded below:
                     x contains a direction along which f is decreasing to -inf, either because f is linear
                     along x and the directional derivative is not zero, or because x is a direction with
                     negative curvature;
                        - 'stopped': the algorithm terminated having exhausted the maximum number of iterations:
                     x is the best solution found so far, but not necessarily the optimal one.
    """

    def __init__(self, f, x, eps=1e-6, max_iter=1000, callback=None, callback_args=(), verbose=False):
        super().__init__(f, x, eps, max_iter, callback, callback_args, verbose)
        if not isinstance(f, Quadratic):
            raise ValueError('f is not a quadratic function')
        if self.x.size != self.f.Q.shape[0]:
            raise ValueError('x size does not match with Q')

    def minimize(self):

        if self.verbose:
            print('iter\tcost\t\tgnorm', end='')
            if self.f.f_star() < np.inf:
                print('\tgap\trate', end='')
                prev_v = np.inf

        while True:
            self.f_x, self.g_x = self.f.function(self.x), self.f.jacobian(self.x)
            ng = np.linalg.norm(self.g_x)

            self.callback()

            if self.is_verbose():
                print('\n{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, self.f_x, ng), end='')
                if self.f.f_star() < np.inf:
                    print('\t{:1.4e}'.format(self.f_x - self.f.f_star()), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((self.f_x - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    else:
                        print('\t\t', end='')
                    prev_v = self.f_x

            # stopping criteria
            if ng <= self.eps:
                self.status = 'optimal'
                break

            if self.iter > self.max_iter:
                self.status = 'stopped'
                break

            d = -self.g_x

            # check if f is unbounded below
            den = d.T.dot(self.f.Q).dot(d)

            if den <= 1e-12:
                # this is actually two different cases:
                #
                # - d.T.dot(Q).dot(d) = 0, i.e., f is linear along d, and since the
                #   gradient is not zero, it is unbounded below;
                #
                # - d.T.dot(Q).dot(d) < 0, i.e., d is a direction of negative curvature
                #   for f, which is then necessarily unbounded below.
                self.status = 'unbounded'
                break

            # compute step size
            a = d.T.dot(d) / den  # or ng ** 2 / den

            # assert np.isclose(d.T.dot(d), ng ** 2)

            step = a * d

            # compute new point
            last_x = self.x + step

            # <\nabla f(x_i), \nabla f(x_i+1)> = 0
            assert np.isclose(self.f.jacobian(self.x).T.dot(self.f.jacobian(last_x)), 0)

            self.x = last_x

            self.iter += 1

        if self.verbose:
            print('\n')

        return self


class SteepestGradientDescent(LineSearchOptimizer):
    """
    Apply the classical Steepest Descent algorithm for the minimization of
    the provided function f.
    # - x is either a [n x 1] real (column) vector denoting the input of
    #   f(), or [] (empty).
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
    #   for the line search. To avoid numerical problems that can occur with
    #   the quadratic interpolation if the derivative at one endpoint is too
    #   large w.r.t. The one at the other (which leads to choosing a point
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
    """

    def __init__(self, f, x, eps=1e-6, max_iter=1000, max_f_eval=1000, m1=0.01, m2=0.9, a_start=1, tau=0.9,
                 sfgrd=0.01, m_inf=-np.inf, min_a=1e-16, callback=None, callback_args=(), verbose=False):
        """

        :param f:          the objective function.
        :param x:          ([n x 1] real column vector): the point where to start the algorithm from.
        :param eps:        (real scalar, optional, default value 1e-6): the accuracy in the stopping
                           criterion: the algorithm is stopped when the norm of the gradient is less
                           than or equal to eps.
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
                                - v (real, scalar): if x == [] this is the best known lower bound on the unconstrained
                                global optimum of f(); it can be -inf if either f() is not bounded below, or no such
                                information is available. If x ~= [] then v = f(x);
                                - g (real, [n x 1] real vector): this also depends on x. If x == [] this is the
                                standard starting point from which the algorithm should start, otherwise it is the
                                gradient of f() at x (or a subgradient if f() is not differentiable at x, which it
                                should not be if you are applying the gradient method to it).
        :return status:    (string): a string describing the status of the algorithm at termination:
                              - 'optimal': the algorithm terminated having proven that x is a(n approximately) optimal
                           solution, i.e., the norm of the gradient at x is less than the required threshold;
                              - 'unbounded': the algorithm has determined an extremely large negative value for f()
                           that is taken as an indication that the problem is unbounded below (a "finite -inf",
                           see m_inf above);
                              - 'stopped': the algorithm terminated having exhausted the maximum number of iterations:
                           x is the bast solution found so far, but not necessarily the optimal one;
                              - 'error': the algorithm found a numerical error that prev_vents it from continuing
                           optimization (see min_a above).
        """
        super().__init__(f, x, eps, max_iter, max_f_eval, m1, m2, a_start, tau, sfgrd,
                         m_inf, min_a, callback, callback_args, verbose)

    def minimize(self):
        last_x = np.zeros(self.f.ndim)  # last point visited in the line search
        last_g = np.zeros(self.f.ndim)  # gradient of last_x
        f_eval = 1  # f() evaluations count ("common" with LSs)

        if self.verbose:
            print('iter\tfeval\tcost\t\tgnorm', end='')
            if self.f.f_star() < np.inf:
                print('\tgap\trate\t', end='')
                prev_v = np.inf
            print('\tls\tit\ta*', end='')

        while True:
            self.f_x, self.g_x = self.f.function(self.x), self.f.jacobian(self.x)
            ng = np.linalg.norm(self.g_x)

            self.callback()

            if self.eps < 0:
                ng0 = -ng  # norm of first subgradient
            else:
                ng0 = 1  # un-scaled stopping criterion

            if self.is_verbose():
                print('\n{:4d}\t{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, f_eval, self.f_x, ng), end='')
                if self.f.f_star() < np.inf:
                    print('\t{:1.4e}'.format(self.f_x - self.f.f_star()), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((self.f_x - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    else:
                        print('\t\t', end='')
                    prev_v = self.f_x

            # stopping criteria
            if ng <= self.eps * ng0:
                self.status = 'optimal'
                break

            if self.iter > self.max_iter or f_eval > self.line_search.max_f_eval:
                self.status = 'stopped'
                break

            d = -self.g_x

            phi_p0 = -ng * ng

            # compute step size
            a, self.f_x, last_x, last_g, f_eval = self.line_search.search(
                d, self.x, last_x, last_g, f_eval, self.f_x, phi_p0, self.is_verbose())

            # output statistics
            if self.is_verbose():
                print('\t{:1.4e}'.format(a), end='')

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
