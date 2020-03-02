import numpy as np

from ml.initializers import glorot_normal, glorot_uniform
from ml.losses import mean_squared_error, cross_entropy
from ml.metrics import mean_euclidean_error, accuracy_score
from ml.neural_network.activations import sigmoid, tanh, relu, linear, softmax
from ml.neural_network.layers import Dense, Conv2D, MaxPool2D, Flatten
from ml.neural_network.neural_network import NeuralNetwork
from ml.regularizers import L1, L2
from optimization.unconstrained.accelerated_gradient import AcceleratedGradient, SteepestDescentAcceleratedGradient
from optimization.unconstrained.adadelta import AdaDelta
from optimization.unconstrained.adagrad import AdaGrad
from optimization.unconstrained.adam import Adam
from optimization.unconstrained.adamax import AdaMax
from optimization.unconstrained.amsgrad import AMSGrad
from optimization.unconstrained.conjugate_gradient import NonlinearConjugateGradient
from optimization.unconstrained.gradient_descent import GradientDescent, SteepestGradientDescent
from optimization.unconstrained.heavy_ball_gradient import HeavyBallGradient
from optimization.unconstrained.newton import Newton
from optimization.unconstrained.quasi_newton import BFGS
from optimization.unconstrained.rmsprop import RMSProp
from optimization.unconstrained.rprop import RProp

line_search_optimizers = [NonlinearConjugateGradient, SteepestDescentAcceleratedGradient, Newton,
                          SteepestGradientDescent, HeavyBallGradient, BFGS]

max_f_eval = [10000, 15000, 20000, 25000, 30000]

epochs = [1000, 500, 200, 100]

stochastic_adaptive_optimizers = [Adam, AdaMax, AMSGrad, AdaGrad, AdaDelta, RProp, RMSProp]

stochastic_optimizers = [GradientDescent, AcceleratedGradient]

momentum = [0.5, 0.6, 0.7, 0.8, 0.9]

activations = [sigmoid, tanh, relu]

learning_rate_epochs = {1000: 0.001,
                        500: 0.01,
                        200: 0.05,
                        100: 0.1}

regularizers = [L1(0.01), L1(0.1), L2(0.01), L2(0.1)]

k_folds = [3, 5]

if __name__ == '__main__':
    ml_cup_train = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X, y = ml_cup_train[:, :-2], ml_cup_train[:, -2:]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20)

    net = NeuralNetwork(
        Dense(20, 20, relu, w_init=glorot_uniform, w_reg=L1(0.1), b_reg=L1(0.1), use_bias=True),
        Dense(20, 20, relu, w_init=glorot_uniform, w_reg=L1(0.1), b_reg=L1(0.1), use_bias=True),
        Dense(20, 2, linear, w_init=glorot_uniform, w_reg=L1(0.1), b_reg=L1(0.1), use_bias=True))

    net.fit(X_train, y_train, loss=mean_squared_error, optimizer=BFGS, learning_rate=0.001, momentum_type='nesterov',
            momentum=0.9, epochs=500, batch_size=None, max_f_eval=2000, verbose=True, plot=True)
    pred = net.predict(X_test)
    print(mean_squared_error(pred, y_test))
    print(mean_euclidean_error(pred, y_test))

    ml_cup_blind = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TS.csv', delimiter=','), 0, 1)

    # mnist = np.load('./ml/data/mnist.npz')
    # X_train, y_train = mnist['x_train'][:, :, :, None], mnist['y_train'][:, None]
    # X_test, y_test = mnist['x_test'][:, :, :, None], mnist['y_test']
    #
    # cnn = NeuralNetwork(
    #     Conv2D(1, 6, (5, 5), (1, 1), 'same', channels_last=True, activation=relu),  # => [n,28,28,6]
    #     MaxPool2D(2, 2),  # => [n, 14, 14, 6]
    #     Conv2D(6, 16, 5, 1, 'same', channels_last=True, activation=relu),  # => [n,14,14,16]
    #     MaxPool2D(2, 2),  # => [n,7,7,16]
    #     Flatten(),  # => [n,7*7*16]
    #     Dense(7 * 7 * 16, 10, softmax))
    # cnn.fit(X_train, y_train, loss=cross_entropy, optimizer=Adam, learning_rate=0.001, momentum_type='nesterov',
    #         momentum=0.9, epochs=300, batch_size=64, max_f_eval=10000, verbose=True, plot=True)
    # pred = cnn.predict(X_test)
    # print(accuracy_score(pred, y_test))
