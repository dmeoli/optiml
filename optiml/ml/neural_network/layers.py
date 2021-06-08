from abc import ABC

import autograd.numpy as np

from .activations import Activation, linear
from .initializers import glorot_uniform
from .regularizers import l2


class Layer(ABC):

    def forward(self, X):
        raise NotImplementedError

    def backward(self, delta):
        raise NotImplementedError


class ParamLayer(Layer, ABC):

    def __init__(self,
                 coef_shape,
                 activation,
                 coef_init,
                 inter_init,
                 coef_reg,
                 inter_reg,
                 fit_intercept,
                 random_state=None):

        if isinstance(activation, Activation):
            self.activation = activation
        else:
            raise TypeError(f'{activation} is not an allowed activation function')

        if coef_init is None:
            self.coef_ = glorot_uniform(coef_shape, random_state=random_state)
        elif callable(coef_init):
            self.coef_ = coef_init(coef_shape, random_state=random_state)
        else:
            self.coef_ = np.asarray(coef_init, dtype=float).reshape(-1, 1)

        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            shape = [1] * len(coef_shape)
            shape[-1] = coef_shape[-1]
            if inter_init is None:
                self.inter_ = np.zeros(shape)
            elif callable(inter_init):
                self.inter_ = inter_init(shape)
            else:
                self.inter_ = np.asarray(inter_init, dtype=float).reshape(-1, 1)

        if coef_reg is None:
            self.coef_reg = l2
        else:
            self.coef_reg = coef_reg

        if inter_reg is None:
            self.inter_reg = l2
        else:
            self.inter_reg = inter_reg


class FullyConnected(ParamLayer):

    def __init__(self,
                 n_in,
                 n_out,
                 activation=linear,
                 coef_init=glorot_uniform,
                 inter_init=np.zeros,
                 coef_reg=l2,
                 inter_reg=l2,
                 fit_intercept=True,
                 random_state=None):
        super(FullyConnected, self).__init__(coef_shape=(n_in, n_out),
                                             activation=activation,
                                             coef_init=coef_init,
                                             inter_init=inter_init,
                                             coef_reg=coef_reg,
                                             inter_reg=inter_reg,
                                             fit_intercept=fit_intercept,
                                             random_state=random_state)
        self.fan_in = n_in
        self.fan_out = n_out

    def forward(self, X):
        self._X = X
        self._WX_b = np.dot(self._X, self.coef_)
        if self.fit_intercept:
            self._WX_b += self.inter_
        return self.activation(self._WX_b)

    def backward(self, delta):
        # dW, db
        dZ = delta * self.activation.jacobian(self._WX_b)
        grads = {'dW': self._X.T.dot(dZ)}
        if self.fit_intercept:
            grads['db'] = np.sum(dZ, axis=0, keepdims=True)
        # dX
        dX = dZ.dot(self.coef_.T)
        return dX, grads
