import autograd.numpy as np
from autograd import jacobian, hessian


class Optimizer:

    def __init__(self, f, x, eps=1e-6, max_iter=1000, callback=None, callback_args=(), verbose=False):
        """

        :param f:        the objective function.
        :param x:        ([n x 1] real column vector): the point where to start the algorithm from.
        :param eps:      (real scalar, optional, default value 1e-6): the accuracy in the stopping
                         criterion: the algorithm is stopped when the norm of the gradient is less
                         than or equal to eps.
        :param max_iter: (integer scalar, optional, default value 1000): the maximum number of iterations.
        :param verbose:  (boolean, optional, default value False): print details about each iteration
                         if True, nothing otherwise.
        """
        if not isinstance(f, OptimizationFunction):
            raise TypeError(f'{f} is not an allowed optimization function')
        self.f = f
        if callable(x):
            self.x = x(f.ndim)
        else:
            self.x = np.asarray(x, dtype=np.float)
        self.f_x = np.nan
        self.g_x = np.zeros(0)
        if self.f.ndim == 2:
            self.x0_history = []
            self.x1_history = []
            self.f_x_history = []
        if not eps > 0:
            raise ValueError('eps must be > 0')
        self.eps = eps
        if not max_iter > 0:
            raise ValueError('max_iter must be > 0')
        self.max_iter = max_iter
        self.iter = 0
        self._callback = callback
        self.callback_args = callback_args
        self.status = 'unknown'
        self.verbose = verbose

    def callback(self, args=None):
        if self.f.ndim == 2:
            self.x0_history.append(self.x[0])
            self.x1_history.append(self.x[1])
            self.f_x_history.append(self.f_x)
        if callable(self._callback):
            self._callback(self, *args, *self.callback_args)

    def is_verbose(self):
        return self.verbose and not self.iter % self.verbose

    def minimize(self):
        raise NotImplementedError


class OptimizationFunction:

    def __init__(self, ndim=2):
        self.jac = jacobian(self.function)
        self.hes = hessian(self.function)
        self.ndim = ndim

    def x_star(self):
        return np.full(fill_value=np.nan, shape=self.ndim)

    def f_star(self):
        return np.inf

    def args(self):
        return ()

    def function(self, x):
        raise NotImplementedError

    def jacobian(self, x):
        """
        The Jacobian (i.e. gradient) of the function.
        :param x: 1D array of points at which the Jacobian is to be computed.
        :return:  the Jacobian of the function at x.
        """
        return self.jac(x)

    def hessian(self, x):
        """
        The Hessian matrix of the function.
        :param x: 1D array of points at which the Hessian is to be computed.
        :return:  the Hessian matrix of the function at x.
        """
        return self.hes(x)
