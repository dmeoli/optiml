import numpy as np

from optimization.functions import Function
from optimization.unconstrained.line_search import AWLS, BLS


class Optimizer:
    def __init__(self, f, wrt=None, eps=1e-6, max_iter=1000, verbose=False, plot=False):
        """

        :param f:        the objective function.
        :param wrt:      ([n x 1] real column vector): the point where to start the algorithm from.
        :param eps:      (real scalar, optional, default value 1e-6): the accuracy in the stopping
                         criterion: the algorithm is stopped when the norm of the gradient is less
                         than or equal to eps.
        :param max_iter: (integer scalar, optional, default value 1000): the maximum number of iterations.
        :param verbose:  (boolean, optional, default value False): print details about each iteration
                         if True, nothing otherwise.
        :param plot:     (boolean, optional, default value False): plot the function's surface and its contours
                         if True and the function's dimension is 2, nothing otherwise.
        """
        if not isinstance(f, Function):
            raise ValueError('f not a function')
        self.f = f
        if wrt is None:
            wrt = np.random.standard_normal(f.n)
        if not np.isrealobj(wrt):
            raise ValueError('x not a real vector')
        self.wrt = np.asarray(wrt)
        self.n = self.wrt.size
        if not np.isscalar(eps):
            raise ValueError('eps is not a real scalar')
        if not eps > 0:
            raise ValueError('eps must be > 0')
        self.eps = eps
        if not np.isscalar(max_iter):
            raise ValueError('max_iter is not an integer scalar')
        self.max_iter = max_iter
        self.iter = 1
        self.verbose = verbose
        self.plot = plot

    def minimize(self):
        return NotImplementedError


class LineSearchOptimizer(Optimizer):
    def __init__(self, f, wrt=None, eps=1e-6, max_f_eval=1000, m1=0.01, m2=0.9, a_start=1, tau=0.9,
                 sfgrd=0.01, m_inf=-np.inf, min_a=1e-16, verbose=False, plot=False):
        """

        :param f:          the objective function.
        :param wrt:        ([n x 1] real column vector): the point where to start the algorithm from.
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
        :param plot:       (boolean, optional, default value False): plot the function's surface and its contours
                           if True and the function's dimension is 2, nothing otherwise.
        """
        super().__init__(f, wrt, eps, verbose=verbose, plot=plot)
        if not np.isscalar(m_inf):
            raise ValueError('m_inf is not a real scalar')
        self.m_inf = m_inf
        if 0 < m2 < 1:
            self.line_search = AWLS(f, max_f_eval, m1, m2, a_start, tau, sfgrd, min_a, verbose)
        else:
            self.line_search = BLS(f, max_f_eval, m1, a_start, min_a, tau, verbose)
