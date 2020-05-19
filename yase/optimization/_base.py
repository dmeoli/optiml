import autograd.numpy as np
from autograd import jacobian, hessian
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm


class Optimizer:

    def __init__(self, f, x, eps=1e-6, max_iter=1000,
                 callback=None, callback_args=(), verbose=False):
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
            raise TypeError('f is not an optimization function')
        self.f = f
        if callable(x):
            self.x = x(f.ndim)
        elif not np.isrealobj(x):
            raise ValueError('x not a real vector')
        else:
            self.x = np.asarray(x, dtype=np.float)
        self.f_x = np.nan
        self.g_x = np.zeros(0)
        if self.f.ndim == 2:
            self.x0_history = []
            self.x1_history = []
            self.f_x_history = []
        if not np.isscalar(eps):
            raise ValueError('eps is not a real scalar')
        if not eps > 0:
            raise ValueError('eps must be > 0')
        self.eps = eps
        if not np.isscalar(max_iter):
            raise ValueError('max_iter is not an integer scalar')
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

    def plot(self, x_min, x_max, y_min, y_max):
        X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = np.array([self.function(np.array([x, y]))
                      for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

        fig = plt.figure(figsize=(16, 8))

        # 3D surface plot
        ax = fig.add_subplot(1, 2, 1, projection='3d', elev=50, azim=-50)
        ax.plot_surface(X, Y, Z, norm=SymLogNorm(linthresh=abs(Z.min()), base=np.e), cmap='jet', alpha=0.5)
        ax.plot([self.x_star()[0]], [self.x_star()[1]], [self.f_star()], marker='*', color='r', markersize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(f'${type(self).__name__}$')

        # 2D contour plot
        ax = fig.add_subplot(1, 2, 2)
        ax.contour(X, Y, Z, 70, cmap='jet', alpha=0.5)
        ax.plot(*self.x_star(), marker='*', color='r', linestyle='None', markersize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        return fig
