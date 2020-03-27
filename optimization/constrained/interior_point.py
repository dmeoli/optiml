import matplotlib.pyplot as plt
import numpy as np

from optimization.optimizer import BoxConstrainedOptimizer
from utils import cholesky_solve


class InteriorPoint(BoxConstrainedOptimizer):
    # Apply the Primal-Dual (feasible) Interior (barrier) Method to the convex
    # Box-Constrained Quadratic program:
    #
    #  (P) min { 1/2 x^T Q x + q^T x : 0 <= x <= ub }
    #
    # - max_iter (integer scalar, optional, default value 1000): the maximum
    #   number of iterations
    #
    # - eps (real scalar, optional, default value 1e-10): the accuracy in the
    #   stopping criterion: the algorithm is stopped when the relative gap
    #   between the value of the current primal and dual feasible solutions is
    #   less than or equal to eps
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

    def __init__(self, f, eps=1e-10, max_iter=1000, verbose=False, plot=False):
        super().__init__(f, eps, max_iter, verbose, plot)

    def minimize(self):

        # the Slackened KKT System for (P) (written without slacks) is:
        #
        #   Q x + q + \lambda^+ - \lambda^- = 0
        #
        #   \lambda^+ (u - x) = \mu e
        #
        #   \lambda^- x = \mu e
        #
        #   0 <= x <= u
        #
        #   \lambda^+ >= 0, \lambda^- >= 0
        #
        # where e is the vector of all ones.
        #
        # if x and (\lambda^+, \lambda^-) satisfy SKKTS, then:
        #
        #   v = 1/2 x^T Q x + q x
        #
        #   p = -\lambda^+ u - 1/2 x^T Q x
        #
        # are, respectively, a valid upper and lower bound on v(P), and:
        #
        #   v - p = \mu 2 n
        #
        # With x -> x + dx, \lambda^+ -> lp + dlp, \lambda^+ -> lm + dlm such that:
        #
        #   Q x + q + lp - lm = 0
        #
        # SKKTS becomes:
        #
        #   Q dx + dlp - dlm = 0                                (1)
        #
        #   (lp + dlp) (u - x - dx) = \mu e                     (2)
        #
        #   (lm + dlm) (x   + dx) = \mu e                       (3)
        #
        #   - x <= dx <= u - x                                  (4)
        #
        #   dlp >= -lp, dlm >= -lp                              (5)
        #
        # inequalities (4) and (5) are just the bounds, and will be taken care of
        # by the appropriate choice of the step size. the well-known trick is
        # linearizing the nonlinear equalities (2) and (3) by just ignoring the
        # bilinear terms, which leads to:
        #
        #   Q dx + dlp - dlm = 0                                (1)
        #
        #  -lp dx + dlp (u - x) = \mu e - lp (u - x)            (2')
        #
        #   lm dx + dlm x = \mu e - lm x                        (3')
        #
        # we can then use (2') and (3') to write:
        #
        #    dlp = (\mu e + lp dx) / (u - x) - lp               (2'')
        #
        #    dlm = (\mu e - lm dx) / x - lm                     (3'')
        #
        # putting (2'') and (3'') in (1) gives:
        #
        #   H dx = w                                            (1')
        #
        # with:
        #
        #   H = Q + lp / (u - x) + lm / x
        #
        #   w = \mu [e / (u - x) - e / x] + lp - lm
        #
        # where note that lp - lm = -Q x - q.
        #
        # The term H - Q is diagonal and strictly positive, hence H is strictly
        # positive definite and nonsingular and (1') has a unique solution.
        #
        # To initialize the algorithm we take x straight in the middle of the box,
        # and then it would be simple to satisfy:
        #
        #   Q x + q + lp - lm = 0
        #
        # by taking lm = [Q x + q]_+ and lp = [-Q x - q]_+. However, by doing
        # so lm and lp would not be interior. The obvious solution is to add to
        # both a term eps * e with some small eps (1e-6)

        # compute a feasible interior dual solution satisfying SKKTS with x for some
        # \mu we don't care much of
        g = self.f.jacobian(self.wrt)
        lp = 1e-6 * np.ones(self.f.n)
        lm = np.copy(lp)
        idx = g >= 0
        lm[idx] = lm[idx] + g[idx]
        idx = np.logical_not(idx)
        lp[idx] = lp[idx] - g[idx]

        if self.verbose:
            print('iter\tv\t\t\tp\t\t\tgap')

        if self.plot and self.n == 2:
            surface_plot, contour_plot, contour_plot, contour_axes = self.f.plot()

        while True:
            v = self.f.function(self.wrt)
            xQx = self.wrt.dot(self.f.Q).dot(self.wrt)
            p = -lp.T.dot(self.f.ub) - 0.5 * xQx
            gap = (v - p) / max(abs(v), 1)

            if self.verbose:
                print('{:4d}\t{:1.4e}\t{:1.4e}\t{:1.4e}'.format(self.iter, v, p, gap))

            # stopping criteria
            if gap <= self.eps:
                status = 'optimal'
                break

            if self.iter >= self.max_iter:
                status = 'stopped'
                break

            # solve the SKKTS
            # note: the "complicated" term in W has the form:
            #
            #  \mu [1 / (u_i - x_i) - 1 / x_i]
            #
            # which can be rewritten:
            #
            #  \mu (u_i - 2 x_i) / [(u_i - x_i) x_i]
            #
            # it appears this last form is *vastly* more numerically stable

            mu = (v - p) / (4 * self.f.n * self.f.n)  # use \rho = 1 / (# of constraints)

            umx = self.f.ub - self.wrt
            H = self.f.Q + np.diag(lp / umx + lm / self.wrt)
            # w = \mu (np.ones(n) / umx - np.ones(n) / self.wrt) + lp - lm
            w = mu * (self.f.ub - 2 * self.wrt) / (umx * self.wrt) + lp - lm

            # and use Cholesky to solve the system
            # because H is symmetric positive definite matrix
            dx = cholesky_solve(H, w)

            dlp = (mu * np.ones(self.f.n) + lp * dx) / umx - lp

            dlm = (mu * np.ones(self.f.n) - lm * dx) / self.wrt - lm

            # compute maximum feasible primal step size
            idx = dx < 0  # negative direction entries
            if any(idx):
                max_t = min(-self.wrt[idx] / dx[idx])
            else:
                max_t = np.inf

            idx = dx > 0  # positive direction entries
            if any(idx):
                max_t = min(max_t, min(umx[idx] / dx[idx]))

            # compute maximum feasible dual step size
            idx = dlp < 0  # negative direction entries
            if any(idx):
                max_t = min(max_t, min(-lp[idx] / dlp[idx]))

            idx = dlm < 0  # negative direction entries
            if any(idx):
                max_t = min(max_t, min(-lm[idx] / dlm[idx]))

            # compute new primal-dual solution

            max_t *= 0.9995  # ensure the new iterate remains interior

            self.wrt += max_t * dx
            lp += max_t * dlp
            lm += max_t * dlm

            # plot the trajectory
            if self.plot and self.n == 2:
                p_xy = np.vstack((self.wrt - max_t * dx, self.wrt)).T
                contour_axes.quiver(p_xy[0, :-1], p_xy[1, :-1], p_xy[0, 1:] - p_xy[0, :-1], p_xy[1, 1:] - p_xy[1, :-1],
                                    scale_units='xy', angles='xy', scale=1, color='k')

            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, v, status
