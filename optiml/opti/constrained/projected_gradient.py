import numpy as np

from optiml.opti.constrained import BoxConstrainedQuadraticOptimizer


class ProjectedGradient(BoxConstrainedQuadraticOptimizer):
    r"""
    Apply the Projected Gradient algorithm with exact line search to the convex
    Box-Constrained Quadratic program

    .. math::

        (P) \quad \min \left\{ \tfrac{1}{2} x^\top Q x + q^\top x : lb \le x \le ub \right\}

    At each iteration the steepest descent direction :math:`d = -\nabla f(x)` is
    projected over the active constraints (its components pushing against an active
    bound are zeroed), and the exact step size minimizing the quadratic along d,

    .. math::

        \alpha = -\frac{d^\top (Q x + q)}{d^\top Q d}

    is taken, clipped to the largest step :math:`\alpha_{\max}` that keeps the iterate
    inside the box :math:`lb \le x \le ub`.
    """

    def __init__(self,
                 quad,
                 ub,
                 lb=None,
                 x=None,
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
        :param eps:           (real scalar, optional, default value 1e-6): the accuracy in the stopping
                              criterion: the algorithm is stopped when the norm of the (projected) gradient
                              is less than or equal to eps.
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
        :return status:       (string): a string describing the status of the algorithm at termination:
                                 - 'optimal': the algorithm terminated having proven that x is a(n approximately)
                              optimal solution, i.e., the norm of the (projected) gradient at x is less than the
                              required threshold;
                                 - 'stopped': the algorithm terminated having exhausted the maximum number of
                              iterations: x is the best solution found so far, but not necessarily the optimal one.
        """
        super(ProjectedGradient, self).__init__(quad=quad,
                                                ub=ub,
                                                lb=lb,
                                                x=x,
                                                eps=eps,
                                                tol=tol,
                                                max_iter=max_iter,
                                                callback=callback,
                                                callback_args=callback_args,
                                                verbose=verbose)

    def minimize(self):

        if self.verbose:
            print('iter\t cost\t\t gnorm', end='')

        while True:
            self.f_x, self.g_x = self.f.function(self.x), self.f.jacobian(self.x)
            d = -self.g_x

            # project the direction over the active constraints
            d[np.logical_and(self.ub - self.x <= 1e-12, d > 0)] = 0
            d[np.logical_and(self.x - self.lb <= 1e-12, d < 0)] = 0

            # compute the norm of the (projected) gradient
            ng = np.linalg.norm(d)

            if self.is_verbose():
                print('\n{:4d}\t{: 1.4e}\t{: 1.4e}'.format(self.iter, self.f_x, ng), end='')

            try:
                self.callback()
            except StopIteration:
                break

            if ng <= self.eps:
                self.status = 'optimal'
                break

            if self.iter >= self.max_iter:
                self.status = 'stopped'
                break

            # first, compute the maximum feasible step size max_t such that:
            #   lb[i] <= x[i] + max_t * d[i] <= ub[i]   for all i

            idx = d > 0  # positive gradient entries
            max_t = min((self.ub[idx] - self.x[idx]) / d[idx], default=np.inf)
            idx = d < 0  # negative gradient entries
            max_t = min(max_t, min((self.lb[idx] - self.x[idx]) / d[idx], default=np.inf))

            # compute optimal unbounded step size:
            #   min { 1/2 (x + a d)^T Q (x + a d) + q^T (x + a d) }
            # min { 1/2 a^2 (d^T Q d) + a d^T (Q x + q) } [ + const ]
            #
            # => a = - d^T (Q x + q) / d^T Q d
            den = d.dot(self.f.Q).dot(d)

            if den <= 1e-16:  # d^T Q d = 0 ==> f is linear along d
                t = max_t  # just take the maximum possible step size
            else:
                # optimal unbounded step size restricted to max feasible step
                t = min(-self.g_x.dot(d) / den, max_t)

            self.x += t * d

            try:
                self.check_lagrangian_dual_optimality()
            except StopIteration:
                break

            self.iter += 1

        self.check_lagrangian_dual_conditions()

        if self.verbose:
            print('\n')

        return self
