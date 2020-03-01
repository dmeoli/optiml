import numpy as np

from ml.initializers import glorot_uniform, zeros
from ml.neural_network.activations import Activation
from ml.regularizers import L2


class Variable:
    def __init__(self, v):
        self.data = v
        self._error = np.empty_like(v)  # for backpropagation of the last layer
        self.info = {}

    def __repr__(self):
        return str(self.data)

    def set_error(self, error):
        assert self._error.shape == error.shape
        self._error[:] = error

    @property
    def error(self):
        return self._error

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim


class Layer:

    def __init__(self, n_in, n_out, activation, w_init, b_init, w_reg, b_reg):
        self.order = None
        self.name = None
        self._x = None
        self.data_vars = {}

        if isinstance(activation, Activation):
            self._a = activation
        else:
            raise TypeError

        if w_init is None:
            self.w = glorot_uniform(n_in, n_out)
        else:
            self.w = w_init(n_in, n_out)

        if b_init is None:
            self.b = zeros((1, n_out))
        else:
            self.b = b_init((1, n_out))

        if w_reg is None:
            self.w_reg = L2()
        else:
            self.w_reg = w_reg

        if b_reg is None:
            self.b_reg = L2()
        else:
            self.b_reg = b_reg

        self._wx_b = None
        self._activated = None

    def forward(self, X):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def _process_input(self, x):
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
            x = Variable(x)
            x.info['new_layer_order'] = 0

        self.data_vars['in'] = x
        # x is Variable, extract _x value from x.data
        self.order = x.info['new_layer_order']
        _x = x.data
        return _x

    def _wrap_out(self, out):
        out = Variable(out)
        out.info['new_layer_order'] = self.order + 1
        self.data_vars['out'] = out
        return out


class Dense(Layer):
    def __init__(self, n_in, n_out, activation, w_init=glorot_uniform, b_init=zeros, w_reg=L2(0.01), b_reg=L2(0.01)):
        super().__init__(n_in, n_out, activation, w_init, b_init, w_reg, b_reg)
        self.fan_in = n_in
        self.fan_out = n_out

    def forward(self, X):
        self._x = self._process_input(X)
        self._wx_b = self._x.dot(self.w) + self.b
        self._activated = self._a.function(self._wx_b)
        wrapped_out = self._wrap_out(self._activated)
        return wrapped_out

    def backward(self):
        # dw, db
        dz = self.data_vars['out'].error
        dz *= self._a.derivative(self._wx_b)
        dw = self._x.T.dot(dz)
        db = np.sum(dz, axis=0, keepdims=True)
        # dx
        self.data_vars['in'].set_error(dz.dot(self.w.T))  # pass error to the layer before
        return dw, db
