import matplotlib.pyplot as plt
import numpy as np

from ml.initializers import random_uniform
from optimization.optimization_function import Quadratic
from optimization.optimizer import Optimizer, LineSearchOptimizer


class SteepestGradientDescentQuadratic(Optimizer):
    """
    Apply the Steepest Gradient Descent algorithm with exact line search to the quadratic function.

        f(x) = 1/2 x^T Q x - q^T x

    :param f:        the objective function.
    :param wrt:      ([n x 1] real column vector): the point where to start the algorithm from
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

    def __init__(self, f, wrt=random_uniform, eps=1e-6, max_iter=1000, verbose=False, plot=False):
        super().__init__(f, wrt, eps=eps, max_iter=max_iter, verbose=verbose, plot=plot)
        if not isinstance(f, Quadratic):
            raise ValueError('f is not a quadratic function')
        if self.wrt.size != self.f.Q.shape[0]:
            raise ValueError('wrt size does not match with Q')

    def minimize(self):
        if self.verbose:
            print('iter\tf(x)\t\t||g(x)||', end='')
            if self.f.f_star() < np.inf:
                print('\tf(x) - f*\trate', end='')
                prev_v = np.inf
            print()

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        for args, kwargs in self.args:
            g = self.f.jacobian(self.wrt, *args, **kwargs)
            ng = np.linalg.norm(g)

            if self.verbose:
                v = self.f.function(self.wrt, *args, **kwargs)
                print('{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, v, ng), end='')
                if self.f.f_star() < np.inf:
                    print('\t{:1.4e}'.format(v - self.f.f_star()), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((v - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    prev_v = v
                print()

            # stopping criteria
            if ng <= self.eps:
                status = 'optimal'
                break

            if self.iter > self.max_iter:
                status = 'stopped'
                break

            d = -g

            # check if f is unbounded below
            den = d.T.dot(self.f.hessian(self.wrt, *args, **kwargs)).dot(d)

            if den <= 1e-12:
                # this is actually two different cases:
                #
                # - d.T.dot(Q).dot(d) = 0, i.e., f is linear along d, and since the
                #   gradient is not zero, it is unbounded below;
                #
                # - d.T.dot(Q).dot(d) < 0, i.e., d is a direction of negative curvature
                #   for f, which is then necessarily unbounded below.
                status = 'unbounded'
                break

            # compute step size
            a = d.T.dot(d) / den  # or ng ** 2 / den

            # assert np.isclose(d.T.dot(d), ng ** 2)

            past_wrt = self.wrt

            # compute new point
            self.wrt += a * d

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((past_wrt, self.wrt)).T
                contour_axes.quiver(p_xy[0, :-1], p_xy[1, :-1], p_xy[0, 1:] - p_xy[0, :-1], p_xy[1, 1:] - p_xy[1, :-1],
                                    scale_units='xy', angles='xy', scale=1, color='k')

            # <\nabla f(x_i), \nabla f(x_i+1)> = 0
            # assert np.isclose(
            #     self.f.jacobian(past_wrt, *args, **kwargs).T.dot(self.f.jacobian(self.wrt, *args, **kwargs)), 0)

            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, status


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

    def __init__(self, f, wrt=random_uniform, batch_size=None, eps=1e-6, max_iter=1000, max_f_eval=1000, m1=0.01,
                 m2=0.9, a_start=1, tau=0.9, sfgrd=0.01, m_inf=-np.inf, min_a=1e-16, verbose=False, plot=False):
        """

        :param f:          the objective function.
        :param wrt:        ([n x 1] real column vector): the point where to start the algorithm from.
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
        :param plot:       (boolean, optional, default value False): plot the function's surface and its contours
                           if True and the function's dimension is 2, nothing otherwise.
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
        super().__init__(f, wrt, batch_size, eps, max_iter, max_f_eval, m1, m2,
                         a_start, tau, sfgrd, m_inf, min_a, verbose, plot)

    def minimize(self):
        last_wrt = np.zeros((self.n,))  # last point visited in the line search
        last_g = np.zeros((self.n,))  # gradient of last_wrt
        f_eval = 1  # f() evaluations count ("common" with LSs)

        if self.verbose:
            if self.f.f_star() < np.inf:
                print('it\t\tf eval\tf(x) - f*\t\t||g(x)||\trate\t', end='')
                prev_v = np.inf
            else:
                print('it\t\tf eval\tf(x)\t\t||g(x)||', end='')
            print('\tls\tit\ta*')

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        for args, kwargs in self.args:
            if self.iter == 1:
                v, g = self.f.function(self.wrt, *args, **kwargs), self.f.jacobian(self.wrt, *args, **kwargs)
                ng = np.linalg.norm(g)

                if self.eps < 0:
                    ng0 = -ng  # norm of first subgradient
                else:
                    ng0 = 1  # un-scaled stopping criterion

            if self.verbose:
                if self.f.f_star() < np.inf:
                    print('{:4d}\t{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, f_eval, (v - self.f.f_star()) /
                                                                  max(abs(self.f.f_star()), 1), ng), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((v - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    else:
                        print('\t\t\t', end='')
                    prev_v = v
                else:
                    print('{:4d}\t{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, f_eval, v, ng), end='')

            # stopping criteria
            if ng <= self.eps * ng0:
                status = 'optimal'
                break

            if self.iter > self.max_iter or f_eval > self.line_search.max_f_eval:
                status = 'stopped'
                break

            d = -g

            phi_p0 = -ng * ng

            # compute step size
            a, v, last_wrt, last_g, f_eval = self.line_search.search(
                d, self.wrt, last_wrt, last_g, f_eval, v, phi_p0, args, kwargs)

            # output statistics
            if self.verbose:
                print('\t{:1.4e}'.format(a))

            if a <= self.line_search.min_a:
                status = 'error'
                break

            if v <= self.m_inf:
                status = 'unbounded'
                break

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((self.wrt, last_wrt)).T
                contour_axes.quiver(p_xy[0, :-1], p_xy[1, :-1], p_xy[0, 1:] - p_xy[0, :-1], p_xy[1, 1:] - p_xy[1, :-1],
                                    scale_units='xy', angles='xy', scale=1, color='k')

            # update new point
            self.wrt = last_wrt

            # update gradient
            g = last_g
            ng = np.linalg.norm(g)

            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, status


class GradientDescent(Optimizer):

    def __init__(self, f, wrt=random_uniform, batch_size=None, eps=1e-6, max_iter=1000, step_rate=0.01,
                 momentum_type='none', momentum=0.9, verbose=False, plot=False):
        super().__init__(f, wrt, batch_size, eps, max_iter, verbose, plot)
        if not np.isscalar(step_rate):
            raise ValueError('step_rate is not a real scalar')
        if not step_rate > 0:
            raise ValueError('step_rate must be > 0')
        self.step_rate = step_rate
        if not np.isscalar(momentum):
            raise ValueError('momentum is not a real scalar')
        if not momentum > 0:
            raise ValueError('momentum must be > 0')
        self.momentum = momentum
        if momentum_type not in ('standard', 'nesterov', 'none'):
            raise ValueError('unknown momentum type {}'.format(momentum_type))
        self.momentum_type = momentum_type
        if momentum_type in ('nesterov', 'standard'):
            self.step = 0

    def minimize(self):
        if self.verbose:
            print('iter\tf(x)\t\t||g(x)||', end='')
            if self.f.f_star() < np.inf:
                print('\tf(x) - f*\trate', end='')
                prev_v = np.inf
            print()

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        for args, kwargs in self.args:
            g = self.f.jacobian(self.wrt, *args, **kwargs)
            ng = np.linalg.norm(g)

            if self.verbose:
                v = self.f.function(self.wrt, *args, **kwargs)
                print('{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, v, ng), end='')
                if self.f.f_star() < np.inf:
                    print('\t{:1.4e}'.format(v - self.f.f_star()), end='')
                    if prev_v < np.inf:
                        print('\t{:1.4e}'.format((v - self.f.f_star()) / (prev_v - self.f.f_star())), end='')
                    prev_v = v
                print()

            # stopping criteria
            if ng <= self.eps:
                status = 'optimal'
                break

            if self.iter > self.max_iter:
                status = 'stopped'
                break

            if self.momentum_type == 'standard':
                step_m1 = self.step
                self.step = self.step_rate * -g + self.momentum * step_m1
                self.wrt += self.step
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                big_jump = self.momentum * step_m1
                self.wrt += big_jump
                g = self.f.jacobian(self.wrt, *args, **kwargs)
                correction = self.step_rate * -g
                self.wrt += correction
                self.step = big_jump + correction
            elif self.momentum_type == 'none':
                self.step = self.step_rate * -g
                self.wrt += self.step

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((self.wrt - self.step, self.wrt)).T
                contour_axes.quiver(p_xy[0, :-1], p_xy[1, :-1], p_xy[0, 1:] - p_xy[0, :-1], p_xy[1, 1:] - p_xy[1, :-1],
                                    scale_units='xy', angles='xy', scale=1, color='k')

            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, status
