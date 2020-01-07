import numpy as np

from ml.activations import Sigmoid
from optimization.test_functions import Function


def cross_entropy_loss(x, y):
    """Cross entropy loss function. x and y are 1D iterable objects."""
    return (-1.0 / len(x)) * sum(_x * np.log(_y) + (1 - _x) * np.log(1 - _y) for _x, _y in zip(x, y))


def mean_squared_error_loss(x, y):
    """Mean squared error loss function. x and y are 1D iterable objects."""
    return (1.0 / len(x)) * sum((_x - _y) ** 2 for _x, _y in zip(x, y))


class LogisticRegression(Function):

    def __init__(self, n_input=5, n_classes=3, n_samples=10):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.pars = np.random.standard_normal(n_input * n_classes + n_classes)
        self.X, self.Z = self.make_data()

    def make_data(self):
        xs = []
        zs = []
        for i in range(self.n_classes):
            x = np.random.standard_normal((self.n_samples, self.n_input))
            # make somehow sure that they are far away from each other
            x += self.n_input * i
            z = np.zeros((self.n_samples, self.n_classes))
            z[:, i] = 1
            xs.append(x)
            zs.append(z)
        X = np.vstack(xs)
        Z = np.vstack(zs)
        return X, Z

    def predict(self, x, input):
        n_weights = self.n_input * self.n_classes
        W = x[:n_weights].reshape((self.n_input, self.n_classes))
        b = x[n_weights:]
        sWXb = Sigmoid().function(np.dot(input, W) + b)
        return sWXb / sWXb.sum(axis=1)[:, np.newaxis]

    def function(self, x, input=None, target=None):
        prediction = self.predict(x, input)
        log_pred = np.log(prediction)
        return -(log_pred * target).mean()

    def jacobian(self, x, input=None, target=None):
        prediction = self.predict(x, input)
        d_f_d_W = np.dot(input.T, target - prediction)
        d_f_d_b = (target - prediction).sum(axis=0)
        d_f_d_all = np.concatenate((d_f_d_W.flatten(), d_f_d_b))
        return d_f_d_all / prediction.shape[0]

    def hessian_product(self, x, p, input=None, target=None):
        eps = 1e-6
        gradient = self.jacobian(x, input, target)
        offset = self.jacobian(x + p * eps, input, target)
        return (offset - gradient) / eps

    def score(self):
        return self.function(self.pars, self.X, self.Z)

    def solved(self, tol=0.1):
        return self.score() - tol < 0
