from abc import ABC

import autograd.numpy as np

from .activations import Activation, linear
from .initializers import glorot_uniform
from .regularizers import l2


class Layer(ABC):
    """
    Base abstract class for all neural network layers. A layer implements
    the ``forward`` pass that maps its input to its output and the
    ``backward`` pass that back-propagates the error signal.
    """

    def forward(self, X):
        raise NotImplementedError

    def backward(self, delta):
        raise NotImplementedError


class ParamLayer(Layer, ABC):
    """
    Base abstract class for all layers with trainable parameters, i.e., a
    coefficient (weight) tensor and, optionally, an intercept (bias) tensor,
    each with its own initializer and regularizer.
    """

    def __init__(self,
                 coef_shape,
                 activation,
                 coef_init,
                 inter_init,
                 coef_reg,
                 inter_reg,
                 fit_intercept,
                 random_state=None):
        """
        Parameters
        ----------

        coef_shape : tuple of int
            Shape of the coefficient (weight) tensor.

        activation : `Activation` instance
            The activation function applied by the layer.

        coef_init : callable, array-like or None
            Initializer for the coefficient tensor. If None, `glorot_uniform`
            is used; if callable, it is called with ``coef_shape`` and
            ``random_state``; otherwise it is used as the initial values.

        inter_init : callable, array-like or None
            Initializer for the intercept tensor. If None, zeros are used;
            if callable, it is called with the intercept shape; otherwise it
            is used as the initial values. Only used when ``fit_intercept`` is True.

        coef_reg : `Regularizer` instance or None
            Regularizer applied to the coefficient tensor. If None, `l2` is used.

        inter_reg : `Regularizer` instance or None
            Regularizer applied to the intercept tensor. If None, `l2` is used.

        fit_intercept : bool
            Whether the layer has an intercept (bias) term.

        random_state : int, RandomState instance or None, default=None
            Controls the pseudo random number generation for the parameters
            initialization.
        """

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
    """
    Fully connected (dense) layer that computes ``activation(X @ W + b)``.
    """

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
        """
        Parameters
        ----------

        n_in : int
            Number of input units (fan-in) of the layer.

        n_out : int
            Number of output units (fan-out) of the layer, i.e., the number
            of neurons.

        activation : `Activation` instance, default=linear
            The activation function applied by the layer.

        coef_init : callable or array-like, default=glorot_uniform
            Initializer for the coefficient (weight) tensor.

        inter_init : callable or array-like, default=np.zeros
            Initializer for the intercept (bias) tensor. Only used when
            ``fit_intercept`` is True.

        coef_reg : `Regularizer` instance, default=l2
            Regularizer applied to the coefficient tensor.

        inter_reg : `Regularizer` instance, default=l2
            Regularizer applied to the intercept tensor.

        fit_intercept : bool, default=True
            Whether to add an intercept (bias) term to the layer.

        random_state : int, RandomState instance or None, default=None
            Controls the pseudo random number generation for the parameters
            initialization.
        """
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
