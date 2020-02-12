import numpy as np

from ml.initializers import GlorotUniform, Initializer, Zeros
from ml.neural_network.activations import Activation
from ml.neural_network.variable import Variable


class Layer:

    def __init__(self):
        self.order = None
        self.name = None
        self._x = None
        self.data_vars = {}

    def forward(self, x):
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
        self.data_vars['out'] = out  # add to layer's data_vars
        return out


class ParamLayer(Layer):
    def __init__(self, w_shape, activation, w_init, b_init, use_bias):
        super().__init__()
        self.param_vars = {}
        self.w = np.empty(w_shape, dtype=np.float32)
        self.param_vars['w'] = self.w
        if use_bias:
            shape = [1] * len(w_shape)
            shape[-1] = w_shape[-1]  # only have bias on the last dimension
            self.b = np.empty(shape, dtype=np.float32)
            self.param_vars['b'] = self.b
        self.use_bias = use_bias

        if isinstance(activation, Activation):
            self._a = activation
        else:
            raise TypeError

        if w_init is None:
            GlorotUniform().initialize(self.w)
        elif isinstance(w_init, Initializer):
            w_init.initialize(self.w)
        else:
            raise TypeError

        if use_bias:
            if b_init is None:
                Zeros().initialize(self.b)
            elif isinstance(b_init, Initializer):
                b_init.initialize(self.b)
            else:
                raise TypeError

        self._wx_b = None
        self._activated = None


class Dense(ParamLayer):
    def __init__(self, n_in, n_out, activation, w_init=None, b_init=None, use_bias=True):
        super().__init__((n_in, n_out), activation, w_init, b_init, use_bias)
        self._n_in = n_in
        self._n_out = n_out

    def forward(self, x):
        self._x = self._process_input(x)
        self._wx_b = self._x.dot(self.w)
        if self.use_bias:
            self._wx_b += self.b
        self._activated = self._a.function(self._wx_b)
        wrapped_out = self._wrap_out(self._activated)
        return wrapped_out

    def backward(self):
        # dw, db
        dz = self.data_vars['out'].error
        dz *= self._a.derivative(self._wx_b)
        grads = {'w': self._x.T.dot(dz)}
        if self.use_bias:
            grads['b'] = np.sum(dz, axis=0, keepdims=True)
        # dx
        self.data_vars['in'].set_error(dz.dot(self.w.T))  # pass error to the layer before
        return grads
