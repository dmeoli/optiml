import numpy as np

from ml.losses import categorical_cross_entropy
from ml.metrics import accuracy_score
from ml.neural_network.activations import relu, softmax
from ml.neural_network.layers import FullyConnected, Conv2D, MaxPool2D, Flatten
from ml.neural_network.neural_network import NeuralNetwork
from optimization.unconstrained.adam import Adam

if __name__ == '__main__':
    mnist = np.load('./ml/data/mnist.npz')
    X_train, y_train = mnist['x_train'][:, :, :, None], mnist['y_train'][:, None]
    X_test, y_test = mnist['x_test'][:, :, :, None], mnist['y_test']

    cnn = NeuralNetwork(
        Conv2D(in_channels=1, out_channels=6, kernel_size=(5, 5), strides=(1, 1),
               padding='same', channels_last=True, activation=relu),  # => [n,28,28,6]
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),  # => [n, 14, 14, 6]
        Conv2D(in_channels=6, out_channels=16, kernel_size=(5, 5), strides=(1, 1),
               padding='same', channels_last=True, activation=relu),  # => [n,14,14,16]
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),  # => [n,7,7,16]
        Flatten(),  # => [n,7*7*16]
        FullyConnected(n_in=7 * 7 * 16, n_out=10, activation=softmax))
    cnn.fit(X_train, y_train, loss=categorical_cross_entropy, optimizer=Adam, learning_rate=0.001,
            momentum_type='nesterov', momentum=0.9, epochs=300, batch_size=64, max_f_eval=10000,
            verbose=True, plot=True)
    pred = cnn.predict(X_test)
    print(accuracy_score(pred, y_test))
