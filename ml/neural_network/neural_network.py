import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelBinarizer

from ml.learning import Learner
from ml.losses import MeanSquaredError
from ml.neural_network.activations import Sigmoid, Softmax
from ml.neural_network.layers import Dense, Layer, ParamLayer
from optimization.optimizer import LineSearchOptimizer
from optimization.unconstrained.adam import Adam


class Network(Layer, Learner):

    def __init__(self, *layers):
        super().__init__()
        self._ordered_layers = []
        self.params = {}
        assert isinstance(layers, (list, tuple))
        for i, l in enumerate(layers):
            assert isinstance(l, Layer)
            self.__setattr__('layer_%i' % i, l)
        self.layers = layers

    def forward(self, x, *args):
        for l in self.layers:
            x = l.forward(x)
        return x.data

    def backward(self, delta):

        # find net order
        layers = []
        for name, v in self.__dict__.items():
            if not isinstance(v, Layer):
                continue
            layer = v
            layer.name = name
            layers.append((layer.order, layer))
        self._ordered_layers = [l[1] for l in sorted(layers, key=lambda x: x[0])]

        # back propagate through this order
        last_layer = self._ordered_layers[-1]
        last_layer.data_vars['out'].set_error(delta)
        for layer in self._ordered_layers[::-1]:
            grads = layer.backward()
            if isinstance(layer, ParamLayer):
                for k in layer.param_vars.keys():
                    self.params[layer.name]['grads'][k][:] = grads[k]

    @property
    def _params(self):
        vars = []
        grads = []
        for layer_p in self.params.values():
            for p_name in layer_p['vars'].keys():
                vars.append(layer_p['vars'][p_name])
                grads.append(layer_p['grads'][p_name])
        return vars, grads

    def fit(self, X, y, loss, optimizer, l_rate=0.01, epochs=100, batch_size=None,
            regularization_type='l1', lmbda=0.1, verbose=False, task='classification'):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        assert task in ('classification', 'regression')
        if task is 'classification':
            lb = LabelBinarizer().fit(y)
            y = lb.transform(y)

        for epoch in range(epochs):
            for var, grad in zip(*self._params):
                loss.jacobian = lambda theta, *args: grad.ravel()

                _loss = loss.function(None, self.forward(X), y)
                self.backward(loss.delta)
                if issubclass(optimizer, LineSearchOptimizer):
                    optimizer(wrt=var.ravel(), step_rate=l_rate, f=loss, max_f_eval=1, batch_size=batch_size).minimize()
                else:
                    optimizer(wrt=var.ravel(), step_rate=l_rate, f=loss, max_iter=1, batch_size=batch_size).minimize()

            if verbose:
                print('Epoch: %i | loss: %.5f | %s: %.2f' %
                      (epoch + 1, _loss, 'acc' if task is 'classification' else 'mse',
                       accuracy_score(lb.inverse_transform(y), self.predict(X))
                       if task is 'classification' else mean_squared_error(y, self.predict(X, task='regression'))))

    def predict(self, X, task='classification'):
        assert task in ('classification', 'regression')
        return np.argmax(self.forward(X).data, axis=1) if task is 'classification' else self.forward(X).data

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, key, value):
        if isinstance(value, ParamLayer):
            layer = value
            self.params[key] = {
                'vars': layer.param_vars,
                'grads': {k: np.empty_like(layer.param_vars[k]) for k in layer.param_vars.keys()}
            }
        object.__setattr__(self, key, value)


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)

    net = Network(Dense(4, 4, Sigmoid()),
                  Dense(4, 4, Sigmoid()),
                  Dense(4, 3, Softmax()))

    net.fit(X, y, loss=MeanSquaredError(X, y), optimizer=Adam, epochs=100, batch_size=None, verbose=True)
    pred = net.predict(X)
    print(pred, '\n', y)
    print(accuracy_score(pred, y))

    # ml_cup = np.delete(np.genfromtxt('../data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    # X, y = ml_cup[:, :-2], ml_cup[:, -2:]
    #
    # net = Network(Dense(20, 20, Tanh()),
    #               Dense(20, 20, Tanh()),
    #               Dense(20, 2, Linear()))
    # net.fit(X, y, loss=MeanSquaredError(X, y), optimizer=Adam, epochs=1000, task='regression', verbose=True)
