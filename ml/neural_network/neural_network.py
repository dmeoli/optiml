import pickle

import numpy as np
from sklearn.datasets import load_iris

from ml.learning import Learner
from ml.neural_network.activations import Sigmoid, Tanh
from ml.neural_network.dataloader import DataLoader
from ml.neural_network.layers import Dense, Conv2D, MaxPool2D, Flatten, Layer, ParamLayer
from ml.neural_network.losses import MSE, SparseSoftMaxCrossEntropyWithLogits, Loss
from ml.neural_network.optimizers import Adam


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

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, loss):
        assert isinstance(loss, Loss)
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
        last_layer.data_vars['out'].set_error(loss.delta)
        for layer in self._ordered_layers[::-1]:
            grads = layer.backward()
            if isinstance(layer, ParamLayer):
                for k in layer.param_vars.keys():
                    self.params[layer.name]['grads'][k][:] = grads[k]

    def fit(self, X, y, loss, optimizer, epochs=100, verbose=True):
        for epoch in range(epochs):
            o = net.forward(X)
            _loss = loss(o, y)
            net.backward(_loss)
            optimizer.step()
            if verbose:
                print("Epoch: %i | loss: %.5f" % (epoch, _loss.data))

    def predict(self, X):
        return np.argmax(net.forward(X).data, axis=1)

    def save(self, path):
        vars = {name: p['vars'] for name, p in self.params.items()}
        with open(path, 'wb') as f:
            pickle.dump(vars, f)

    def restore(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        for name, param in params.items():
            for p_name in self.params[name]['vars'].keys():
                self.params[name]['vars'][p_name][:] = param[p_name]
                self.params[name]['vars'][p_name][:] = param[p_name]

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

    # IRIS DATASET
    X, y = load_iris(return_X_y=True)
    X, y = X, y[:, np.newaxis]

    net = Network(Dense(4, 4, Tanh()),
                  Dense(4, 4, Tanh()),
                  Dense(4, 2, Sigmoid()))

    net.fit(X, y, loss=MSE(), optimizer=Adam(net.params, l_rate=0.1), epochs=30)
    print(net.predict(X), '\n', y.ravel())

    # ML CUP 2019 DATASET
    ml_cup = np.delete(np.genfromtxt('../data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X, y = ml_cup[:, :-2], ml_cup[:, -2:]

    net = Network(Dense(20, 20, Tanh()),
                  Dense(20, 20, Tanh()),
                  Dense(20, 2, Sigmoid()))
    net.fit(X, y, loss=MSE(), optimizer=Adam(net.params, l_rate=0.1), epochs=100)

    # MNIST DATASET
    f = np.load('../data/mnist.npz')
    train_x, train_y = f['x_train'][:, :, :, None], f['y_train'][:, None]
    test_x, test_y = f['x_test'][:, :, :, None], f['y_test']

    # from keras.datasets import mnist
    #
    # (train_x, train_y), (test_x, test_y) = mnist.load_data()

    cnn = Network(Conv2D(1, 6, (5, 5), (1, 1), 'same', channels_last=True),  # => [n,28,28,6]
                  MaxPool2D(2, 2),  # => [n, 14, 14, 6]
                  Conv2D(6, 16, 5, 1, 'same', channels_last=True),  # => [n,14,14,16]
                  MaxPool2D(2, 2),  # => [n,7,7,16]
                  Flatten(),  # => [n,7*7*16]
                  Dense(7 * 7 * 16, 10))
    opt = Adam(cnn.params, 0.001)
    loss_fn = SparseSoftMaxCrossEntropyWithLogits()

    train_loader = DataLoader(train_x, train_y, batch_size=64)
    for epoch in range(300):
        bx, by = train_loader.next_batch()
        by_ = cnn.forward(bx)
        loss = loss_fn(by_, by)
        cnn.backward(loss)
        opt.step()
        if epoch % 50 == 0:
            # ty_ = cnn.forward(test_x)
            # acc = accuracy(np.argmax(ty_.data, axis=1), test_y)
            print("Epoch: %i | loss: %.3f | acc: %.2f" % (epoch, loss.data, 0.0))
