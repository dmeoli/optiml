import numpy as np

from ml.initializers import glorot_uniform
from ml.losses import mean_squared_error
from ml.metrics import mean_euclidean_error
from ml.neural_network.activations import sigmoid, tanh, relu, linear
from ml.neural_network.layers import FullyConnected
from ml.neural_network.neural_network import NeuralNetwork
from ml.regularizers import L2, L1
from optimization.unconstrained.accelerated_gradient import AcceleratedGradient, SteepestDescentAcceleratedGradient
from optimization.unconstrained.adadelta import AdaDelta
from optimization.unconstrained.adagrad import AdaGrad
from optimization.unconstrained.adam import Adam
from optimization.unconstrained.adamax import AdaMax
from optimization.unconstrained.amsgrad import AMSGrad
from optimization.unconstrained.conjugate_gradient import NonlinearConjugateGradient
from optimization.unconstrained.gradient_descent import GradientDescent, SteepestGradientDescent
from optimization.unconstrained.heavy_ball_gradient import HeavyBallGradient
from optimization.unconstrained.proximal_bundle import ProximalBundle
from optimization.unconstrained.quasi_newton import BFGS
from optimization.unconstrained.rmsprop import RMSProp
from optimization.unconstrained.rprop import RProp

line_search_optimizers = [NonlinearConjugateGradient, SteepestDescentAcceleratedGradient,
                          SteepestGradientDescent, HeavyBallGradient, BFGS]

max_f_eval = [10000, 15000, 20000, 25000, 30000]

epochs = [1000, 500, 200, 100]

stochastic_adaptive_optimizers = [Adam, AdaMax, AMSGrad, AdaGrad, AdaDelta, RProp, RMSProp]

stochastic_optimizers = [GradientDescent, AcceleratedGradient]

others = [ProximalBundle]

momentum = [0.5, 0.6, 0.7, 0.8, 0.9]

activations = [sigmoid, tanh, relu]

learning_rate = [0.001, 0.003, 0.006, 0.009, 0.01, 0.03, 0.06, 0.09, 0.1, 0.3, 0.6, 0.9]

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

    net = NeuralNetwork(
        FullyConnected(20, 20, sigmoid, w_init=glorot_uniform, w_reg=L1(0.1), b_reg=L1(0.1), use_bias=True),
        FullyConnected(20, 20, sigmoid, w_init=glorot_uniform, w_reg=L1(0.1), b_reg=L1(0.1), use_bias=True),
        FullyConnected(20, 2, linear, w_init=glorot_uniform, w_reg=L1(0.1), b_reg=L1(0.1), use_bias=True))

    net.fit(X_train, y_train, loss=mean_squared_error, optimizer=BFGS, learning_rate=0.01, momentum_type='nesterov',
            momentum=0.9, epochs=1000, batch_size=None, max_f_eval=25000, verbose=True, plot=True)
    pred = net.predict(X_test)
    print(mean_squared_error(pred, y_test))
    print(mean_euclidean_error(pred, y_test))

    ml_cup_blind = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TS.csv', delimiter=','), 0, 1)
