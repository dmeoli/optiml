import matplotlib.pyplot as plt
import numpy as np

from optimization.constrained.projected_gradient import ConstrainedOptimizer
from optimization.optimizer import LineSearchOptimizer


class LagrangianDual(LineSearchOptimizer, ConstrainedOptimizer):
    # Solve the convex Box-Constrained Quadratic program:
    #
    #  (P) min { 1/2 x^T Q x - q^T x : 0 <= x <= ub }
    #
    # The box constraints 0 <= x <= u are relaxed (with Lagrangian multipliers
    # \lambda^- and \lambda^+, respectively) and the corresponding Lagrangian
    # dual is solved by means of an ad-hoc implementation of the Projected
    # Gradient method (since the Lagrangian multipliers are constrained in
    # sign, and the Lagrangian function is differentiable owing to
    # strict-positive-definiteness of Q) using a classical Armijo Wolfe line search.
    #
    # A rough Lagrangian heuristic is implemented whereby the dual solution is
    # projected on the box at each iteration to provide an upper bound, which
    # is then used in the stopping criterion.
    #
    # - dolh (logical scalar, optional, default value 1): true if a quick
    #   Lagrangian Heuristic is implemented whereby the current solution of the
    #   Lagrangian subproblem is projected on the box constraints
    #
    # - eps (real scalar, optional, default value 1e-6): the accuracy in the
    #   stopping criterion. This depends on the dolh parameter, see above: if
    #   dolh = true then the algorithm is stopped when the relative gap
    #   between the value of the best dual solution (the current one) and the
    #   value of the best upper bound obtained so far (by means of the
    #   heuristic) is less than or equal to eps, otherwise the stopping
    #   criterion is on the (relative, w.r.t. the first one) norm of the
    #   (projected) gradient
    #
    # - max_f_eval (integer scalar, optional, default value 1000): the maximum
    #   number of function evaluations (hence, iterations will be not more than
    #   max_f_eval because at each iteration at least a function evaluation is
    #   performed, possibly more due to the line search).
    #
    # - m1 (real scalar, optional, default value 0.01): first parameter of the
    #   Armijo-Wolfe-type line search (sufficient decrease). Has to be in (0,1)
    #
    # - m2 (real scalar, optional, default value 0.9): the second parameter of
    #   the Armijo-Wolfe-type line search (strong curvature condition), it
    #   should to be in (0,1);
    #
    # - a_start (real scalar, optional, default value 1): starting value of
    #   alpha in the line search (> 0)
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
    # - min_a (real scalar, optional, default value 1e-16): if the algorithm
    #   determines a step size value <= mina, this is taken as an indication
    #   that something has gone wrong (the gradient is not a direction of
    #   descent, so maybe the function is not differentiable) and computation
    #   is stopped. It is legal to take mina = 0, thereby in fact skipping this
    #   test.
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
    #
    #   = 'error': the algorithm found a numerical error that prevents it from
    #     continuing optimization (see mina above)
    #
    # - l ([2 * n x 1] real column vector, optional): the best Lagrangian
    #   multipliers found so far (possibly the optimal ones)

    def __init__(self, f, dolh=True, eps=1e-6, max_iter=1000, max_f_eval=1000, m1=0.01, m2=0.9, a_start=1,
                 tau=0.95, sfgrd=0.01, m_inf=-np.inf, min_a=1e-12, verbose=False, plot=False):
        super().__init__(f, None, None, eps, max_iter, max_f_eval, m1, m2,
                         a_start, tau, sfgrd, m_inf, min_a, verbose, plot)
        # compute the Cholesky factorization of Q, this will be used
        # at each iteration to solve the Lagrangian relaxation
        try:
            self.R = np.linalg.cholesky(self.f.hessian(self.wrt))
        except np.linalg.LinAlgError:
            raise ValueError('Q is not positive definite, this is not yet supported')
        self.dolh = dolh
        self.f_eval = 1  # f() evaluations count ("common" with LSs)

    def minimize(self, ub):
        self.ub = ub

        self.wrt = ub / 2  # initial feasible solution is the middle of the box

        last_wrt = np.zeros((self.n,))  # last point visited in the line search

        v = self.f.function(self.wrt)

        if self.dolh:
            print('iter\tf eval\tub\t\tp(l)\t\tgap', end='')
        else:
            print('iter\tf eval\tp(l)\t\t||g(x)||', end='')
        print('\tls\tit\ta*')

        _lambda = np.zeros(2 * self.f.n)
        p, last_g, v = self.phi(v, _lambda)

        while True:
            # project the direction = -gradient over the active constraints
            d = -last_g
            d[_lambda <= np.logical_and(1e-12, d < 0)] = 0

            if self.dolh:
                # compute the relative gap
                gap = (v + p) / max(abs(v), 1)

                print('{:4d}\t{:4d}\t{:1.4e}\t{:1.4e}\t{:1.4e}\t'.format(self.iter, self.f_eval, v, -p, gap))

                if gap <= self.eps:
                    status = 'optimal'
                    break
            else:
                # compute the norm of the projected gradient
                ng = np.linalg.norm(d)

                print('{:4d}\t{:4d}\t{:1.8e}\t{:1.4e}\t'.format(self.iter, self.f_eval, -p, ng))

                if self.f_eval is 1:
                    ng0 = ng
                if ng <= self.eps * ng0:
                    status = 'optimal'
                    break

            if self.f_eval >= self.line_search.max_f_eval:
                status = 'stopped'
                break

            # first, compute the maximum feasible step size max_t such that:
            #
            #   0 <= lambda[i] + max_t * d[i]   for all i

            idx = d < 0  # negative gradient entries
            if any(idx):
                max_t = min(self.line_search.a_start, min(-_lambda(idx) / d(idx)))
            else:
                max_t = self.line_search.a_start

            # now run the line search
            phi_p0 = last_g.T * d

            # compute step size
            self.line_search.a_start = max_t
            a, v, last_wrt, last_g, self.f_eval = self.line_search.search(
                d, self.wrt, last_wrt, last_g, self.f_eval, p, phi_p0)

            print('\t{:1.4e}'.format(a))

            if a <= self.line_search.min_a:
                status = 'error'
                break

            _lambda = _lambda + a * d

            self.iter += 1

        if self.verbose:
            print()
        if self.plot and self.n == 2:
            plt.show()
        return self.wrt, status, _lambda

    def solve_lagrangian(self, lmbda=None):
        # The Lagrangian relaxation of the problem is:
        #
        #    min { (1/2) x^T Q x + q^T x - lambda^+ (u - x) - lambda^- x
        #  min { (1/2) x^T Q x + (q^T + lambda^+ - lambda^-) x - lambda^+ u
        #
        # where lambda^+ are the first n components of lmbda, and lambda^- the
        # last n components; both are constrained to be >= 0.
        #
        # The optimal solution of the Lagrangian relaxation is the (unique)
        # solution of the linear system:
        #
        #       Q x = -q - lambda^+ + lambda^-
        #
        # Since we have computed at the beginning the Cholesky factorization of Q,
        # i.e., Q = R^T R, where R is upper triangular and therefore R^T is lower
        # triangular, we obtain this by just two triangular backsolves:
        #
        #       R^T z = -q - lambda^+ + lambda^-
        #
        #       R x = z
        #
        # return the function value and the primal solution.

        ql = self.f.q + lmbda[:self.f.n] - lmbda[self.f.n:]
        from scipy.linalg import lu_factor, lu_solve
        # TODO solve the system with LDL^T Cholesky indefinite factorization or with null space method
        z = lu_solve(lu_factor(self.f.hessian(self.wrt).T), -ql)
        y = lu_solve(lu_factor(self.f.hessian(self.wrt)), z)
        # z = solve_triangular(self.R.T, -ql, lower=True)
        # y = solve_triangular(self.R, z, lower=False)

        # compute phi-value
        p = (0.5 * y.T.dot(self.f.hessian(self.wrt)) + ql.T).dot(y) - lmbda[:self.f.n].T.dot(self.ub)

        self.f_eval += 1

        return p, y

    def phi(self, v, lmbda=None):
        # Compute the Lagrangian function of the problem. With x the
        # optimal solution of the minimization problem (see solve_lagrangian), the
        # gradient at lambda is [x - u; -x].
        # However, the line search is written for minimization but we rather want
        # to maximize phi(), hence we have to change the sign of both function
        # values and gradient entries.

        # solve the Lagrangian relaxation
        p, y = self.solve_lagrangian(lmbda)
        p = -p

        # compute gradient
        g = np.vstack((self.ub - y, y))

        if self.dolh:
            # compute an heuristic solution out of the solution y of the Lagrangian
            # relaxation by projecting y on the box

            y[y < 0] = 0
            idx = y > self.ub
            y[idx] = self.ub[idx]

            # compute cost of feasible solution
            pv = 0.5 * y.T.dot(self.f.hessian(self.wrt)).dot(y) + self.f.q.T.dot(y)

            if pv < v:  # it is better than best one found so far
                self.wrt = y  # y becomes the incumbent
                v = pv

        return p, g, v
