import numpy as np
from cvxpy import Variable, Problem, Minimize, sum_squares

from .. import Optimizer


class ProximalBundle(Optimizer):
    # Apply the Proximal Bundle Method for the minimization of the provided
    # function f.
    #
    # Input:
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
    #   start, otherwise it is a subgradient of f() at x (possibly the
    #   gradient, but you should not apply this algorithm to a differentiable
    #   f)
    #
    # The other [optional] input parameters are:
    #
    # - x (either [n x 1] real vector or [], default []): starting point.
    #   If x == [], the default starting point provided by f() is used.
    #
    # - mu (real scalar, optional, default value 1): the fixed weight to be
    #   given to the stabilizing term throughout all the algorithm. It must
    #   be a strictly positive number.
    #
    # - m1 (real scalar, optional, default value 0.01): parameter of the
    #   Armijo-like condition used to declare a Serious Step; has to be in
    #   [0,1).
    #
    # - eps (real scalar, optional, default value 1e-6): the accuracy in the
    #   stopping criterion: the algorithm is stopped when the norm of the
    #   direction (optimal solution of the master problem) is less than or
    #   equal to mu * eps. If a negative value is provided, this is used in a
    #   *relative* stopping criterion: the algorithm is stopped when the norm
    #   of the direction is less than or equal to
    #      mu * (-eps) * || norm of the first gradient ||.
    #
    # - m_inf (real scalar, optional, default value -inf): if the algorithm
    #   determines a value for f() <= m_inf this is taken as an indication that
    #   the problem is unbounded below and computation is stopped
    #   (a "finite -inf").
    #
    # Output:
    #
    # - x ([n x 1] real column vector): the best solution found so far.
    #
    # - status (string): a string describing the status of the algorithm at
    #   termination
    #
    #   = 'optimal': the algorithm terminated having proven that x is an
    #     (approximately) optimal solution; this only happens when "cheating",
    #     i.e., explicitly uses v_* = f([]) > -inf, unless in the very
    #     unlikely case that f() spontaneously produces an almost-null
    #     subgradient
    #
    #   = 'unbounded': the algorithm has determined an extremely large negative
    #     value for f() that is taken as an indication that the problem is
    #     unbounded below (a "finite -inf", see m_inf above)
    #
    #   = 'stopped': the algorithm terminated having exhausted the maximum
    #     number of iterations: x is the bast solution found so far, but not
    #     necessarily the optimal one

    def __init__(self, f, x, mu=1, m1=0.01, eps=1e-6, max_iter=1000, m_inf=-np.inf, master_solver='ecos', momentum=0.9,
                 momentum_type='none', callback=None, callback_args=(), verbose=False, master_verbose=False):
        super().__init__(f, x, eps, max_iter, callback, callback_args, verbose)
        if not np.isscalar(mu):
            raise ValueError('mu is not a real scalar')
        if not mu > 0:
            raise ValueError('mu must be > 0')
        self.mu = mu
        if not np.isscalar(m1):
            raise ValueError('m1 is not a real scalar')
        if not 0 < m1 < 1:
            raise ValueError('m1 is not in (0,1)')
        self.m1 = m1
        if not np.isscalar(m_inf):
            raise ValueError('m_inf is not a real scalar')
        self.m_inf = m_inf
        if not np.isscalar(momentum):
            raise ValueError('momentum is not a real scalar')
        if not momentum > 0:
            raise ValueError('momentum must be > 0')
        self.momentum = momentum
        if momentum_type not in ('standard', 'nesterov', 'none'):
            raise ValueError(f'unknown momentum type {momentum_type}')
        self.momentum_type = momentum_type
        self.master_solver = master_solver
        self.master_verbose = master_verbose
        self.step = 0

    def minimize(self):

        if self.verbose:
            if self.f.f_star() < np.inf:
                print('iter\tcost\t\tgap\tdnorm', end='')
            else:
                print('iter\tcost\t\tdnorm', end='')

        while True:

            if self.iter == 0:
                # compute first function and subgradient
                self.f_x, self.g_x = self.f.function(self.x), self.f.jacobian(self.x)

                G = self.g_x.T  # matrix of subgradients
                F = self.f_x - self.g_x.T.dot(self.x)  # vector of translated function values
                # each (fxi , gi , xi) gives the constraint
                #
                #  v >= fxi + gi' * (x + d - xi) = gi' * (x + d) + (fi - gi' * xi)
                #
                # so we just keep the single constant fi - gi' * xi instead of xi

                ng = np.linalg.norm(self.g_x)
                if self.eps < 0:
                    ng0 = -ng  # norm of first subgradient
                else:
                    ng0 = 1  # un-scaled stopping criterion

            # construct the master problem
            d = Variable((self.f.ndim, 1))
            v = Variable(1)

            M = [v >= F + G * (self.x.reshape(-1, 1) + d)]

            if self.f.f_star() < np.inf:
                # cheating: use information about f_star in the model
                M = M + [v >= self.f.f_star()]

            # objective function
            c = v + self.mu * sum_squares(d) / 2

            # solve the master problem
            Problem(Minimize(c), M).solve(solver=self.master_solver.upper(), verbose=self.master_verbose)

            d = -d.value.ravel()
            v = v.value.item()

            nd = np.linalg.norm(d)

            # output statistics
            if self.is_verbose():
                if self.f.f_star() < np.inf:
                    print('\n{:4d}\t{:1.4e}\t{:1.4e}\t{:1.4e}'.format(
                        self.iter, self.f_x, (self.f_x - self.f.f_star()) / max(abs(self.f.f_star()), 1), nd), end='')
                else:
                    print('\n{:4d}\t{:1.4e}\t{:1.4e}'.format(self.iter, self.f_x, nd), end='')

            # stopping criteria
            if self.mu * nd <= self.eps * ng0:
                self.status = 'optimal'
                break

            if self.iter >= self.max_iter:
                self.status = 'stopped'
                break

            if self.momentum_type == 'standard':
                step_m1 = self.step
                step1 = self.momentum * step_m1
            elif self.momentum_type == 'nesterov':
                step_m1 = self.step
                step1 = self.momentum * step_m1
                self.x -= step1

            last_x = self.x - (step1 + d if self.momentum_type == 'standard' else d)

            # compute function and subgradient
            fd, self.g_x = self.f.function(last_x), self.f.jacobian(last_x)

            if fd <= self.m_inf:
                self.status = 'unbounded'
                break

            G = np.vstack((G, self.g_x.T))
            F = np.vstack((F, fd - self.g_x.T.dot(last_x)))

            if fd <= self.f_x + self.m1 * (v - self.f_x):
                self.x = last_x
                self.step = d if self.momentum_type == 'none' else step1 + d
                self.f_x = fd
            else:
                self.step = 0

            self.iter += 1

            self.callback()

        if self.verbose:
            print('\n')
        return self
