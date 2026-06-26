import numpy as np
from cvxpy import Variable, Problem, Minimize, sum_squares

from .. import Optimizer


class ProximalBundle(Optimizer):
    """
    Apply the Proximal Bundle Method for the minimization of the provided
    function f, which is assumed to be (possibly) nondifferentiable.

    At each iteration a cutting-plane model of f is built from the subgradients
    collected so far and a stabilized master problem, regularized by a proximal
    term weighted by mu, is solved to compute the search direction; the candidate
    point is then accepted as a Serious Step or rejected as a Null Step according
    to an Armijo-like sufficient decrease condition.
    """

    def __init__(self,
                 f,
                 x=None,
                 mu=1,
                 m1=0.01,
                 eps=1e-6,
                 tol=1e-8,
                 max_iter=1000,
                 m_inf=-np.inf,
                 callback=None,
                 callback_args=(),
                 master_solver='clarabel',
                 master_verbose=False,
                 random_state=None,
                 verbose=False):
        """

        :param f:              the objective function.
        :param x:              ([n x 1] real column vector, optional): the point where to start the
                               algorithm from; if not provided, the default starting point provided
                               by f() is used.
        :param mu:             (real scalar, optional, default value 1): the fixed weight to be given to
                               the stabilizing term throughout all the algorithm. It must be a strictly
                               positive number.
        :param m1:             (real scalar, optional, default value 0.01): parameter of the Armijo-like
                               condition used to declare a Serious Step; has to be in (0,1).
        :param eps:            (real scalar, optional, default value 1e-6): the accuracy in the stopping
                               criterion: the algorithm is stopped when the norm of the direction (optimal
                               solution of the master problem) is less than or equal to mu * eps. If a
                               negative value is provided, this is used in a *relative* stopping criterion:
                               the algorithm is stopped when the norm of the direction is less than or equal
                               to mu * (- eps) * || norm of the first gradient ||.
        :param tol:            (real scalar, optional, default value 1e-8): the tolerance used to check the
                               optimality conditions when f is a Lagrangian dual relaxation.
        :param max_iter:       (integer scalar, optional, default value 1000): the maximum number of iterations.
        :param m_inf:          (real scalar, optional, default value -inf): if the algorithm determines a value
                               for f() <= m_inf this is taken as an indication that the problem is unbounded
                               below and computation is stopped (a "finite -inf").
        :param callback:       (callable, optional, default value None): a function called at each iteration
                               with the optimizer instance as first argument.
        :param callback_args:  (tuple, optional, default value ()): additional arguments passed to callback.
        :param master_solver:  (string, optional, default value 'clarabel'): the cvxpy solver used to solve the
                               master problem at each iteration.
        :param master_verbose: (boolean, optional, default value False): print details about the resolution
                               of the master problem if True, nothing otherwise.
        :param random_state:   (integer scalar, optional, default value None): seed for the random number
                               generator used to initialize the starting point when x is not provided.
        :param verbose:        (boolean, optional, default value False): print details about each iteration
                               if True, nothing otherwise.
        :return x:             ([n x 1] real column vector): the best solution found so far.
        :return status:        (string): a string describing the status of the algorithm at termination:
                                  - 'optimal': the algorithm terminated having proven that x is a(n approximately)
                               optimal solution; this only happens when "cheating", i.e., explicitly uses
                               v_* = f([]) > -inf, unless in the very unlikely case that f() spontaneously
                               produces an almost-null subgradient;
                                  - 'unbounded': the algorithm has determined an extremely large negative value
                               for f() that is taken as an indication that the problem is unbounded below
                               (a "finite -inf", see m_inf above);
                                  - 'stopped': the algorithm terminated having exhausted the maximum number of
                               iterations: x is the best solution found so far, but not necessarily the optimal one;
                                  - 'error': the master problem could not be solved at some iteration.
        """
        super(ProximalBundle, self).__init__(f=f,
                                             x=x,
                                             eps=eps,
                                             tol=tol,
                                             max_iter=max_iter,
                                             callback=callback,
                                             callback_args=callback_args,
                                             random_state=random_state,
                                             verbose=verbose)
        if not mu > 0:
            raise ValueError('mu must be > 0')
        self.mu = mu
        if not 0 < m1 < 1:
            raise ValueError('m1 has to lie in (0,1)')
        self.m1 = m1
        self.m_inf = m_inf
        self.master_solver = master_solver
        self.master_verbose = master_verbose
        if self.f.ndim <= 3:
            self.x0_history_ns = []
            self.x1_history_ns = []
            self.f_x_history_ns = []

    def minimize(self):

        if self.verbose:
            print('iter\t cost\t\t dnorm', end='')
            if self.f.f_star() < np.inf:
                print('\t\t gap', end='')

        # compute first function and subgradient
        self.f_x, self.g_x = self.f.function(self.x), self.f.jacobian(self.x)

        ng = np.linalg.norm(self.g_x)
        if self.eps < 0:
            ng0 = -ng  # norm of first subgradient
        else:
            ng0 = 1  # un-scaled stopping criterion

        G = self.g_x  # matrix of subgradients

        F = self.f_x - self.g_x.dot(self.x)  # vector of translated function values

        while True:

            # construct the master problem
            d = Variable(self.x.size)
            v = Variable(1)

            # each (fxi , gi , xi) gives the constraint:
            #
            #  v >= fxi + gi^T * (x + d - xi) = gi^T * (x + d) + (fi - gi^T * xi)
            #
            # so we just keep the single constant fi - gi^T * xi instead of xi
            M = [v >= F + G @ (self.x + d)]

            if self.f.f_star() < np.inf:
                # cheating: use information about f_star in the model
                M += [v >= self.f.f_star()]

            # objective function
            c = v + self.mu * sum_squares(d) / 2

            if self.is_verbose() and self.master_verbose:
                print('\n')

            # solve the master problem
            Problem(Minimize(c), M).solve(solver=self.master_solver.upper(),
                                          verbose=self.is_verbose() and self.master_verbose)

            try:
                d = -d.value.ravel()
            except AttributeError:
                self.status = 'error'
                break

            v = v.value.item()

            self.nd = np.linalg.norm(d)

            # output statistics
            if self.is_verbose():
                print('\n{:4d}\t{: 1.4e}\t{: 1.4e}'.format(self.iter, self.f_x, self.nd), end='')
                if self.f.f_star() < np.inf:
                    print('\t{: 1.4e}'.format((self.f_x - self.f.f_star()) /
                                              max(abs(self.f.f_star()), 1)), end='')

            # stopping criteria
            if self.mu * self.nd <= self.eps * ng0:
                self.status = 'optimal'
                break

            if self.iter >= self.max_iter:
                self.status = 'stopped'
                break

            last_x = self.x - d

            # compute function and subgradient
            fd, self.g_x = self.f.function(last_x), self.f.jacobian(last_x)

            try:
                self.callback()
            except StopIteration:
                break

            if fd <= self.m_inf:
                self.status = 'unbounded'
                break

            G = np.vstack((G, self.g_x))

            F = np.hstack((F, fd - self.g_x.dot(last_x)))

            if fd <= self.f_x + self.m1 * (v - self.f_x):  # SS: serious step

                self.x = last_x
                self.f_x = fd

                try:
                    self.check_lagrangian_dual_optimality()
                except StopIteration:
                    break

            else:  # NS: null step

                if self.f.ndim <= 3:
                    self.x0_history_ns.append(self.x[0])
                    self.x1_history_ns.append(self.x[1])
                    self.f_x_history_ns.append(self.f_x)

            self.iter += 1

        self.check_lagrangian_dual_conditions()

        if self.verbose:
            print('\n')

        return self
