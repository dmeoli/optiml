import numpy as np

from . import BoxConstrainedQuadraticOptimizer


class ProjectedGradient(BoxConstrainedQuadraticOptimizer):
    # Apply the Projected Gradient algorithm with exact line search to the
    # convex Box-Constrained Quadratic program:
    #
    #  (P) min { 1/2 x^T Q x + q^T x : 0 <= x <= ub }
    #
    # - eps (real scalar, optional, default value 1e-6): the accuracy in the
    #   stopping criterion: the algorithm is stopped when the norm of the
    #   (projected) gradient is less than or equal to eps
    #
    # - max_iter (integer scalar, optional, default value 1000): the maximum
    #   number of iterations
    #
    # Output:
    #
    # - v (real scalar): the best function value found so far (possibly the
    #   optimal one)
    #
    # - x ([n x 1] real column vector, optional): the best solution found so
    #   far (possibly the optimal one)
    #
    # - status (string, optional): a string describing the status of the
    #   algorithm at termination, with the following possible values:
    #
    #   = 'optimal': the algorithm terminated having proven that x is an
    #     (approximately) optimal solution, i.e., the norm of the gradient at x
    #     is less than the required threshold
    #
    #   = 'stopped': the algorithm terminated having exhausted the maximum
    #     number of iterations: x is the bast solution found so far, but not
    #     necessarily the optimal one

    def __init__(self, f, eps=1e-6, max_iter=1000, callback=None, callback_args=(), verbose=False):
        super().__init__(f, eps, max_iter, callback, callback_args, verbose)

    def minimize(self):

        if self.verbose:
            print('iter\tcost\t\tgnorm')

        while True:
            self.f_x, self.g_x = self.f.function(self.x), self.f.jacobian(self.x)
            d = -self.g_x

            # project the direction over the active constraints
            d[np.logical_and(self.f.ub - self.x <= 1e-12, d > 0)] = 0
            d[np.logical_and(self.x <= 1e-12, d < 0)] = 0

            # compute the norm of the (projected) gradient
            ng = np.linalg.norm(d)

            if self.is_verbose():
                print('{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, self.f_x, ng))

            self.callback()

            if ng <= self.eps:
                self.status = 'optimal'
                break

            if self.iter >= self.max_iter:
                self.status = 'stopped'
                break

            # first, compute the maximum feasible step size max_t such that:
            #   0 <= x[i] + max_t d[i] <= ub[i]   for all i

            idx = d > 0  # positive gradient entries
            max_t = min((self.f.ub[idx] - self.x[idx]) / d[idx], default=np.inf)
            idx = d < 0  # negative gradient entries
            max_t = min(max_t, min(-self.x[idx] / d[idx], default=np.inf))

            # compute optimal unbounded step size:
            #   min { 1/2 (x + a d)^T Q (x + a d) + q^T (x + a d) }
            # min { 1/2 a^2 (d^T Q d) + a d^T (Q x + q) } [ + const ]
            #
            # => a = - d^T (Q x + q) / d^T Q d
            den = d.T.dot(self.f.Q).dot(d)

            if den <= 1e-16:  # d^T Q d = 0 ==> f is linear along d
                t = max_t  # just take the maximum possible step size
            else:
                # optimal unbounded step size restricted to max feasible step
                t = min(-self.g_x.T.dot(d) / den, max_t)

            self.x += t * d

            self.iter += 1

        if self.verbose:
            print()

        return self
