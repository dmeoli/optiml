import numpy as np

from optiml.opti.constrained import BoxConstrainedQuadraticOptimizer


class FrankWolfe(BoxConstrainedQuadraticOptimizer):
    r"""
    Apply the (possibly, stabilized) Frank-Wolfe algorithm with exact line search
    to the convex Box-Constrained Quadratic program

    .. math::

        (P) \quad \min \left\{ \tfrac{1}{2} x^\top Q x + q^\top x : lb \le x \le ub \right\}

    At each iteration the linearized problem :math:`\min \{ \nabla f(x)^\top y : lb \le y \le ub \}`
    is solved over the box, whose vertex picks, per coordinate, :math:`ub_i` where the
    gradient is negative and :math:`lb_i` otherwise; this yields the descent direction
    :math:`d = y - x`, the lower bound :math:`f(x) + \nabla f(x)^\top (y - x)` (whose gap
    to f(x) drives the stopping criterion), and the exact step size

    .. math::

        \alpha = \min \left( -\frac{\nabla f(x)^\top d}{d^\top Q d},\; 1 \right)

    Optionally the search is stabilized by restricting y to a box of relative size t
    around the current point.
    """

    def __init__(self,
                 quad,
                 ub,
                 lb=None,
                 x=None,
                 t=0.,
                 eps=1e-6,
                 tol=1e-8,
                 max_iter=1000,
                 callback=None,
                 callback_args=(),
                 verbose=False):
        r"""

        :param quad:          the quadratic function :math:`\tfrac{1}{2} x^\top Q x + q^\top x` to be minimized.
        :param ub:            ([n x 1] real column vector): the upper bound of the box, i.e., the
                              variables are constrained to lie in :math:`lb \le x \le ub`.
        :param lb:            ([n x 1] real column vector, optional): the lower bound of the box;
                              if not provided it defaults to the all-zeros vector.
        :param x:             ([n x 1] real column vector, optional): the point where to start the
                              algorithm from; if not provided, it starts from the middle of the box.
        :param t:             (real scalar, optional, default value 0): if the stabilized version of the
                              approach is used, then the new point is chosen in the box of relative size
                              around the current point, i.e., the component x[i] is allowed to change by
                              not more than plus or minus t * ub[i]. If t = 0, then the non-stabilized
                              version of the algorithm is used. It has to lie in [0, 1).
        :param eps:           (real scalar, optional, default value 1e-6): the accuracy in the stopping
                              criterion: the algorithm is stopped when the relative gap between the value
                              of the best primal solution (the current one) and the value of the best lower
                              bound obtained so far is less than or equal to eps.
        :param tol:           (real scalar, optional, default value 1e-8): the tolerance used to check the
                              optimality conditions when f is a Lagrangian dual relaxation.
        :param max_iter:      (integer scalar, optional, default value 1000): the maximum number of iterations.
        :param callback:      (callable, optional, default value None): a function called at each iteration
                              with the optimizer instance as first argument.
        :param callback_args: (tuple, optional, default value ()): additional arguments passed to callback.
        :param verbose:       (boolean, optional, default value False): print details about each iteration
                              if True, nothing otherwise.
        :return x:            ([n x 1] real column vector): the best solution found so far (possibly the
                              optimal one).
        :return status:       (string): the status of the algorithm at termination, one of:
                              ``optimal`` (x is a(n approximately) optimal solution) or
                              ``stopped`` (the maximum number of iterations or evaluations
                              was reached).
        """
        super(FrankWolfe, self).__init__(quad=quad,
                                         ub=ub,
                                         lb=lb,
                                         x=x,
                                         eps=eps,
                                         tol=tol,
                                         max_iter=max_iter,
                                         callback=callback,
                                         callback_args=callback_args,
                                         verbose=verbose)
        if not 0 <= t < 1:
            raise ValueError('t has to lie in [0, 1)')
        self.t = t

    def minimize(self):

        best_lb = -np.inf  # best lower bound so far (= none, really)

        if self.verbose:
            print('iter\t cost\t\t lb\t\t gap', end='')

        while True:
            self.f_x, self.g_x = self.f.function(self.x), self.f.jacobian(self.x)

            # solve min { <g, y> : lb <= y <= ub }, whose vertex picks, per coordinate,
            # the upper bound where the gradient is negative and the lower bound otherwise
            y = np.where(self.g_x < 0, self.ub, self.lb)

            # compute the lower bound: remember that the first-order approximation
            # is f(x) + g(y - x)
            lb = self.f_x + self.g_x.dot(y - self.x)
            if lb > best_lb:
                best_lb = lb

            # compute the relative gap
            gap = (self.f_x - best_lb) / max(abs(self.f_x), 1)

            if self.is_verbose():
                print('\n{:4d}\t{: 1.4e}\t{: 1.4e}\t{: 1.4e}'.format(self.iter, self.f_x, best_lb, gap), end='')

            try:
                self.callback()
            except StopIteration:
                break

            if gap <= self.eps:
                self.status = 'optimal'
                break

            if self.iter >= self.max_iter:
                self.status = 'stopped'
                break

            # in the stabilized case, restrict y to a box of relative size t around x
            if self.t > 0:
                radius = self.t * (self.ub - self.lb)
                y = np.clip(y, self.x - radius, self.x + radius)

            # compute step size
            # we are taking direction d = y - x and y is feasible, hence the
            # maximum step size is 1
            d = y - self.x

            # compute optimal unbounded step size:
            #   min 1/2 (x + a d)^T * Q * (x + a d) + q^T * (x + a d)
            # min 1/2 a^2 (d^T * Q * d) + a d^T * (Q * x + q) [+ const]
            #
            # ==> a = -d^T * (Q * x + q) / d^T * Q * d
            #
            den = d.dot(self.f.Q).dot(d)

            if den <= 1e-16:  # d^T * Q * d = 0  ==>  f is linear along d
                a = 1  # just take the maximum possible step size
            else:
                # optimal unbounded step size restricted to max feasible step
                a = min(-self.g_x.dot(d) / den, 1)

            self.x += a * d

            try:
                self.check_lagrangian_dual_optimality()
            except StopIteration:
                break

            self.iter += 1

        self.check_lagrangian_dual_conditions()

        if self.verbose:
            print('\n')

        return self
