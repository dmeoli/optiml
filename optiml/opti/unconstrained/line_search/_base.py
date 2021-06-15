from abc import ABC

import numpy as np

from ... import Optimizer
from .line_search import ArmijoWolfeLineSearch, BacktrackingLineSearch


class LineSearchOptimizer(Optimizer, ABC):

    def __init__(self,
                 f,
                 x=None,
                 eps=1e-6,
                 tol=1e-8,
                 max_iter=1000,
                 max_f_eval=1000,
                 m1=0.01,
                 m2=0.9,
                 a_start=1,
                 tau=0.9,
                 sfgrd=0.01,
                 m_inf=-np.inf,
                 min_a=1e-16,
                 callback=None,
                 callback_args=(),
                 random_state=None,
                 verbose=False):
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
                           [as * (1 + sfgrd) , am * (1 - sfgrd)], being [as , am] the current interval, whatever
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
        """
        super(LineSearchOptimizer, self).__init__(f=f,
                                                  x=x,
                                                  eps=eps,
                                                  tol=tol,
                                                  max_iter=max_iter,
                                                  callback=callback,
                                                  callback_args=callback_args,
                                                  random_state=random_state,
                                                  verbose=verbose)
        self.ng = 0
        self.m_inf = m_inf
        if 0 < m2 < 1:
            self.line_search = ArmijoWolfeLineSearch(f, max_f_eval, m1, m2, a_start, tau, sfgrd, min_a)
        else:
            self.line_search = BacktrackingLineSearch(f, max_f_eval, m1, a_start, min_a, tau)
        self.f_eval = 1

    def _print_header(self):
        if self.verbose:
            print('iter\tfeval\t cost\t\t gnorm', end='')
            if self.f.f_star() < np.inf:
                print('\t\t gap\t\t rate', end='')
                self.prev_f_x = np.inf

    def _print_info(self):
        if self.is_verbose():
            print('\n{:4d}\t{:5d}\t{: 1.4e}\t{: 1.4e}'.format(self.iter, self.f_eval, self.f_x, self.ng), end='')
            if self.f.f_star() < np.inf:
                print('\t{: 1.4e}'.format((self.f_x - self.f.f_star()) /
                                          max(abs(self.f.f_star()), 1)), end='')
                if self.prev_f_x < np.inf:
                    print('\t{: 1.4e}'.format((self.f_x - self.f.f_star()) /
                                              max(abs(self.prev_f_x - self.f.f_star()), 1)), end='')
                else:
                    print('\t\t', end='')
                self.prev_f_x = self.f_x
