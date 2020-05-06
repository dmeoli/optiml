from optimization.optimization_function import BoxConstrained
from optimization.optimizer import Optimizer


class BoxConstrainedOptimizer(Optimizer):
    def __init__(self, f, eps=1e-6, max_iter=1000, callback=None, callback_args=(), verbose=False):
        if not isinstance(f, BoxConstrained):
            raise TypeError('f is not a box-constrained quadratic function')
        super().__init__(f, f.ub / 2,  # starts from the middle of the box
                         eps, max_iter, callback, callback_args, verbose)

    def minimize(self):
        raise NotImplementedError
