from random import random

import matplotlib.pyplot as plt
import numpy as np

from ml.neural_network import BackPropagation, get_batch, init_examples
from optimization_test_functions import Rosenbrock, gen_quad_2
from optimizer import Optimizer, LineSearchOptimizer
from unconstrained.line_search import armijo_wolfe_line_search, backtracking_line_search
from utils import vector_add, scalar_vector_product


class SteepestGradientDescentQuadratic(Optimizer):
    """
    Apply the Steepest Gradient Descent algorithm with exact line search to the quadratic function.

        f(x) = 1/2 x^T Q x - q^T x

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

    def __init__(self, f, x, f_star=np.inf, eps=1e-6, max_iter=1000, verbose=False, plot=False):
        super().__init__(f, x, eps, max_iter, verbose, plot)
        if self.x.size != self.f.hessian().shape[0]:
            raise ValueError('x size does not match with Q')
        if not np.isrealobj(f_star) or not np.isscalar(f_star):
            raise ValueError('f_star is not a real scalar')
        self.f_star = f_star

    def minimize(self):
        if self.verbose:
            print('iter\tf(x)\t\t\t||nabla f(x)||', end='')
        if self.f_star < np.inf:
            if self.verbose:
                print('\tf(x) - f*\trate', end='')
            prev_v = np.inf
        if self.verbose:
            print()

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        i = 1
        while True:
            # compute function value and gradient
            v, g = self.f.function(self.x), self.f.jacobian(self.x)
            ng = np.linalg.norm(g)

            # output statistics
            if self.verbose:
                print('{:4d}\t{:1.8e}\t\t{:1.4e}'.format(i, v, ng), end='')
            if self.f_star < np.inf:
                if self.verbose:
                    print('\t{:1.4e}'.format(v - self.f_star), end='')
                if prev_v < np.inf:
                    if self.verbose:
                        print('\t{:1.4e}'.format((v - self.f_star) / (prev_v - self.f_star)), end='')
                prev_v = v
            if self.verbose:
                print()

            # stopping criteria
            if ng <= self.eps:
                status = 'optimal'
                break

            if i > self.max_iter:
                status = 'stopped'
                break

            # check if f is unbounded below
            den = g.T.dot(self.f.hessian()).dot(g)

            if den <= 1e-12:
                # this is actually two different cases:
                #
                # - g.T.dot(Q).dot(g) = 0, i.e., f is linear along g, and since the
                #   gradient is not zero, it is unbounded below;
                #
                # - g.T.dot(Q).dot(g) < 0, i.e., g is a direction of negative curvature
                #   for f, which is then necessarily unbounded below.
                status = 'unbounded'
                break

            # compute step size
            a = g.T.dot(g) / den  # or ng ** 2 / den

            # assert np.isclose(g.T.dot(g), ng ** 2)

            # compute new point
            new_x = self.x - a * g

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.hstack((self.x, new_x))
                contour_axes.plot(p_xy[0], p_xy[1], color='k')

            # <\nabla f(x_i), \nabla f(x_i+1)> = 0
            # assert np.isclose(self.f.jacobian(self.x).T.dot(self.f.jacobian(self.x - a * g)), 0)

            self.x = new_x
            i += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.x, status


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

    def __init__(self, f, x, eps=1e-6, max_f_eval=1000, m1=0.01, m2=0.9, a_start=1, tau=0.9,
                 sfgrd=0.01, m_inf=-np.inf, min_a=1e-16, verbose=False, plot=False):
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
        super().__init__(f, x, eps, max_f_eval, m1, m2, a_start, tau, sfgrd, m_inf, min_a, verbose, plot)

    def minimize(self):
        f_star = self.f.function([])

        last_x = np.zeros((self.n, 1))  # last point visited in the line search
        last_g = np.zeros((self.n, 1))  # gradient of last_x
        f_eval = 1  # f() evaluations count ("common" with LSs)

        if f_star > -np.inf:
            if self.verbose:
                print('f_eval\trel gap\t\t|| g(x) ||\t\trate\t', end='')
            prev_v = np.inf
        else:
            if self.verbose:
                print('f_eval\tf(x)\t\t\t|| g(x) ||\t', end='')
        if self.verbose:
            print('ls f_eval\ta*')

        v, g = self.f.function(self.x), self.f.jacobian(self.x)
        ng = np.linalg.norm(g)
        if self.eps < 0:
            ng0 = -ng  # norm of first subgradient
        else:
            ng0 = 1  # un-scaled stopping criterion

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        while True:
            # output statistics
            if f_star > -np.inf:
                if self.verbose:
                    print('{:4d}\t{:1.4e}\t{:1.4e}'.format(f_eval, (v - f_star) / max(abs(f_star), 1), ng), end='')
                if prev_v < np.inf:
                    if self.verbose:
                        print('\t{:1.4e}'.format((v - f_star) / (prev_v - f_star)), end='')
                else:
                    if self.verbose:
                        print('\t\t\t', end='')
                prev_v = v
            else:
                if self.verbose:
                    print('{:4d}\t{:1.8e}\t\t{:1.4e}'.format(f_eval, v, ng), end='')

            # stopping criteria
            if ng <= self.eps * ng0:
                status = 'optimal'
                break

            if f_eval > self.max_f_eval:
                status = 'stopped'
                break

            d = -g

            phi_p0 = -ng * ng

            # compute step size
            if 0 < self.m2 < 1:
                a, v, last_x, last_g, _, f_eval = \
                    armijo_wolfe_line_search(self.f, d, self.x, last_x, last_g, None, f_eval, self.max_f_eval,
                                             self.min_a, self.sfgrd, v, phi_p0, self.a_start, self.m1, self.m2,
                                             self.tau, self.verbose)
            else:
                a, v, last_x, last_g, _, f_eval = \
                    backtracking_line_search(self.f, d, self.x, last_x, last_g, None, f_eval, self.max_f_eval,
                                             self.min_a, v, phi_p0, self.a_start, self.m1, self.tau, self.verbose)

            # output statistics
            if self.verbose:
                print('\t\t{:1.4e}'.format(a))

            if a <= self.min_a:
                status = 'error'
                break

            if v <= self.m_inf:
                status = 'unbounded'
                break

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.hstack((self.x, last_x))
                contour_axes.plot(p_xy[0], p_xy[1], color='k')

            # update new point
            self.x = last_x

            # update gradient
            g = last_g
            ng = np.linalg.norm(g)

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.x, status


class GradientDescent(Optimizer):
    """
    Apply the classical Stochastic Gradient Descent algorithm for the minimization
    of the provided function f.

    Gradient descent works by iteratively performing updates solely based on
    the first derivative of a problem. The gradient is calculated and multiplied
    with a scalar (or component wise with a vector) to do a step in the problem
    space. For speed ups, a technique called "momentum" is often used, which
    averages search steps over iterations.

    Even though gradient descent is pretty simple it can be very effective if
    well tuned (in terms of its hyper parameters step rate and momentum).
    Sometimes the use of schedules for both parameters is necessary. See
    ``schedule`` for basic schedules.

    Gradient descent is also very robust to stochasticity in the objective
    function. This might result from noise injected into it (e.g. in the case
    of denoising auto encoders) or because it is based on data samples (e.g. in
    the case of stochastic mini batches.)

    Given a step rate :math:`\\alpha` and a function :math:`f'` to evaluate the
    search direction the current parameters :math:`\\theta_t` the following
    update is performed:

    .. math::
        v_{t+1} &= \\alpha f'(\\theta_t) \\\\
        \\theta_{t+1} &= \\theta_t - v_{t+1}.

    If we also have a momentum :math:`\\beta` and are using standard momentum,
    we update the parameters according to:

    .. math::
        v_{t+1} &= \\alpha f'(\\theta_t) + \\beta v_{t} \\\\
        \\theta_{t+1} &= \\theta_t - v_{t+1}

    In some cases (e.g. learning the parameters of deep networks), using
    Nesterov momentum can be beneficial. In this case, we first make a momentum
    step and then evaluate the gradient at the location in between. Thus,
    there is an additional cost of an addition of the parameters.

    .. math::
        \\theta_{t+{1 \\over 2}} &= \\theta_t - \\beta v_t \\\\
        v_{t+1} &= \\alpha f'(\\theta_{t + {1 \\over 2}}) \\\\
        \\theta_{t+1} &= \\theta_t - v_{t+1}

    which can be specified additionally by the initialization argument
    ``momentum_type``.


    Attributes
    ----------
    f : Callable
        First derivative of the objective function. Returns an array of the \
        same shape as ``.wrt``.

    x : array_like
    Current solution to the problem. Can be given as a first argument to \
    ``.fprime``.

    step_rate : float or array_like
        Step rate to multiply the gradients with.

    momentum : float or array_like
        Momentum to multiply previous steps with.

    momentum_type : string (either "standard" or "nesterov")
        When to add the momentum term to the parameter vector; in the first \
        case it will be done after the calculation of the gradient, in the\
        latter before.
    """

    def __init__(self, f, x, eps=1e-6, max_iter=1000, step_rate=0.1, momentum=0.0,
                 momentum_type='none', verbose=False, plot=False, args=None):
        """Create a GradientDescent object.

        Parameters
        ----------
        x : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``fprime`` should accept this array as a first argument.

        fprime : callable
            Callable that given a solution vector as first parameter and *args
            and **kwargs drawn from the iterations ``args`` returns a
            search direction, such as a gradient.

        step_rate : float or array_like, or iterable of that
            Step rate to use during optimization. Can be given as a single
            scalar value or as an array for a different step rate of each
            parameter of the problem.

            Can also be given as an iterator; in that case, every iteration
            of the optimization takes a new element as a step rate from that
            iterator.

        momentum : float or array_like, or iterable of that
          Momentum to use during optimization. Can be specified analogously
          (but independent of) step rate.

        momentum_type : string (either "standard" or "nesterov")
            When to add the momentum term to the parameter vector; in the first
            case it will be done after the calculation of the gradient, in the
            latter before.

        args : iterable
            Iterator of arguments which ``fprime`` will be called
            with.
        """

        super().__init__(f, x, verbose, plot, args)
        self.step_rate = step_rate
        self.momentum = momentum
        if momentum_type not in ('nesterov', 'standard', 'none'):
            raise ValueError('unknown momentum type')
        self.momentum_type = momentum_type
        self.step = 0

    def __iter__(self):
        for args, kwargs in self.args:
            step_rate = self.step_rate
            momentum = self.momentum
            step_m1 = self.step

            if self.momentum_type == 'standard':
                gradient = self.f.jacobian(self.x, *args, **kwargs)
                step = gradient * step_rate + momentum * step_m1
                self.x -= step
            elif self.momentum_type == 'nesterov':
                big_jump = momentum * step_m1
                self.x -= big_jump
                gradient = self.f.jacobian(self.x, *args, **kwargs)
                correction = step_rate * gradient
                self.x -= correction
                step = big_jump + correction
            elif self.momentum_type == 'none':
                gradient = self.f.jacobian(self.x, *args, **kwargs)
                step = step_m1 + gradient * step_rate
                self.x -= step

            self.step = step
            self.n_iter += 1
            yield self.extended_info(gradient=gradient, args=args, kwargs=kwargs)


if __name__ == "__main__":
    print(SteepestGradientDescentQuadratic(gen_quad_2, [[-1], [1]], f_star=gen_quad_2.function([]),
                                           verbose=True, plot=True).minimize())
    print()
    print(SteepestGradientDescent(Rosenbrock(), [[-1], [1]], max_f_eval=10000, verbose=True, plot=True).minimize())
    print()
    print(GradientDescent(Rosenbrock(), [[-1], [1]], step_rate=0.01, verbose=True, plot=True))
    print()
    print(GradientDescent(Rosenbrock(), [[-1], [1]], step_rate=0.01, momentum=0.9,
                          momentum_type='standard', verbose=True, plot=True))
    print()
    print(GradientDescent(Rosenbrock(), [[-1], [1]], step_rate=0.01, momentum=0.9,
                          momentum_type='nesterov', verbose=True, plot=True))


def gradient_descent(x, inputs, target, net, loss, epochs=1000, l_rate=0.01, batch_size=1, verbose=None):
    """
    Gradient descent algorithm to update the learnable parameters of a network.
    :return: the updated network
    """

    for e in range(epochs):
        total_loss = 0
        random.shuffle(x)
        weights = [[node.weights for node in layer.nodes] for layer in net]

        for batch in get_batch(x, batch_size):
            inputs, targets = init_examples(batch, inputs, target, len(net[-1].nodes))
            # compute gradients of weights
            gs, batch_loss = BackPropagation(inputs, targets, weights, net, loss)
            # update weights with gradient descent
            weights = vector_add(weights, scalar_vector_product(-l_rate, gs))
            total_loss += batch_loss
            # update the weights of network each batch
            for i in range(len(net)):
                if weights[i]:
                    for j in range(len(weights[i])):
                        net[i].nodes[j].weights = weights[i][j]

        if verbose and (e + 1) % verbose == 0:
            print("epoch:{}, total_loss:{}".format(e + 1, total_loss))

    return net
