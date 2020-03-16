import matplotlib.pyplot as plt
import numpy as np

from optimization.constrained.projected_gradient import ConstrainedOptimizer


class FrankWolfe(ConstrainedOptimizer):
    # Apply the (possibly, stabilized) Frank-Wolfe algorithm with exact line
    # search to the convex Box-Constrained Quadratic program
    #
    #  (P) min { 1/2 x^T Q x - q^T x : Ax = b, 0 <= x <= ub }
    #
    # Input:
    #
    # - BCQP, the structure encoding the BCQP to be solved within its fields:
    #
    #   = BCQP.Q:  n \times n symmetric positive semidefinite real matrix
    #
    #   = BCQP.q:  n \times 1 real vector
    #
    #   = BCQP.ub: n \times 1 real vector > 0
    #
    # - eps (real scalar, optional, default value 1e-6): the accuracy in the
    #   stopping criterion: the algorithm is stopped when the relative gap
    #   between the value of the best primal solution (the current one) and the
    #   value of the best lower bound obtained so far is less than or equal to
    #   eps
    #
    # - max_iter (integer scalar, optional, default value 1000): the maximum
    #   number of iterations
    #
    # - t (real scalar scalar, optional, default value 0): if the stabilized
    #   version of the approach is used, then the new point is chosen in the
    #   box of relative size around the current point, i.e., the component
    #   x[i] is allowed to change by not more than plus or minus t * u[i].
    #   if t = 0, then the non-stabilized version of the algorithm is used.
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

    def __init__(self, f, t=0., eps=1e-6, max_iter=1000, verbose=False, plot=False):
        super().__init__(f, eps, max_iter, verbose, plot)
        if not np.isreal(t) or not np.isscalar(t):
            raise ValueError('t is not a real scalar')
        if not 0 <= t < 1:
            raise ValueError('t has to lie in [0, 1)')
        self.t = t

    def minimize(self, A, b, ub):

        self.wrt = ub / 2  # start from the middle of the box

        best_lb = -np.inf  # best lower bound so far (= none, really)

        print('iter\tf(x)\t\tlb\t\t\tgap')

        while True:
            v, g = self.f.function(self.wrt), self.f.jacobian(self.wrt)

            # solve min { <g, y> : 0 <= y <= u }
            y = np.zeros(self.f.n)
            idx = g < 0
            y[idx] = ub[idx]

            # compute the lower bound: remember that the first-order approximation
            # is f( x ) + g ( y - x )
            lb = v + g.T.dot(y - self.wrt)
            if lb > best_lb:
                best_lb = lb

            # compute the relative gap
            gap = (v - best_lb) / max(abs(v), 1)

            print('{:4d}\t{:1.4e}\t{:1.4e}\t{:1.4e}'.format(self.iter, v, best_lb, gap))

            if gap <= self.eps:
                status = 'optimal'
                break

            if self.iter >= self.max_iter:
                status = 'stopped'
                break

            # in the stabilized case, restrict y in the box
            if self.t > 0:
                y = max(self.wrt - self.t * ub, min(self.wrt + self.t * ub, y))

            # compute step size
            # we are taking direction d = y - x and y is feasible, hence the
            # maximum step size is 1
            d = y - self.wrt

            # compute optimal unbounded step size:
            # min (1/2) (x + a d)^T * Q * (x + a d) + q^T * (x + a d) =
            #     (1/2) a^2 (d^T * Q * d) + a d^T * (Q * x + q) [+ const]
            #
            # ==> a = -d^T * (Q * x + q) / d^T * Q * d
            #
            den = d.T.dot(self.f.hessian(self.wrt)).dot(d)

            if den <= 1e-16:  # d^T * Q * d = 0  ==>  f is linear along d
                alpha = 1  # just take the maximum possible step size
            else:
                # optimal unbounded step size restricted to max feasible step
                alpha = min(-g.T.dot(d) / den, 1)

            self.wrt += alpha * d

            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, status
