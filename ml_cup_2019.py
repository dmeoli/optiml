import numpy as np

from ml.initializers import glorot_normal, glorot_uniform
from ml.losses import mean_squared_error
from ml.metrics import mean_euclidean_error
from ml.neural_network.activations import sigmoid, tanh, relu, linear
from ml.neural_network.layers import Dense
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

max_f_eval = [1000, 1500, 2000, 2500, 3000]

stochastic_adaptive_optimizers = [Adam, AdaMax, AMSGrad, AdaGrad, AdaDelta, RProp, RMSProp]

stochastic_optimizers = [GradientDescent, AcceleratedGradient]

activations = [sigmoid, tanh, relu]

learning_rate_epochs = {1000: 0.001,
                        500: 0.01,
                        200: 0.05,
                        100: 0.1}

momentum = [0.5, 0.6, 0.7, 0.8, 0.9]

k_folds = [1, 2, 3, 4, 5]

initializers = [glorot_normal, glorot_uniform]

if __name__ == '__main__':
    # X, y = load_iris(return_X_y=True)

    # net = NeuralNetwork(Dense(4, 4, sigmoid),
    #                     Dense(4, 4, sigmoid),
    #                     Dense(4, 3, softmax))
    #
    # net.fit(X, y, loss=cross_entropy, optimizer=GradientDescent, learning_rate=0.01, momentum_type='none',
    #         momentum=0.9, regularizer=L2(0.01), epochs=300, batch_size=None, verbose=True, plot=True)
    # pred = net.predict(X)
    # print(pred)
    # print(accuracy_score(pred, y))

    ml_cup_train = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X, y = ml_cup_train[:, :-2], ml_cup_train[:, -2:]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    # from sklearn.preprocessing import StandardScaler
    #
    # feature_scaler = StandardScaler()
    # X_train = feature_scaler.fit_transform(X_train)
    # X_test = feature_scaler.transform(X_test)

    net = NeuralNetwork(Dense(20, 20, sigmoid, w_init=glorot_uniform, w_reg=L1(0.11), b_reg=L1(0.11)),
                        Dense(20, 20, sigmoid, w_init=glorot_uniform, w_reg=L1(0.11), b_reg=L1(0.11)),
                        Dense(20, 2, linear, w_init=glorot_uniform, w_reg=L1(0.11), b_reg=L1(0.11)))

    net.fit(X_train, y_train, loss=mean_squared_error, optimizer=BFGS, learning_rate=0.001,
            momentum_type='nesterov', momentum=0.9, regularizer=L1(0.01), epochs=1000,
            batch_size=None, verbose=True, plot=True)
    pred = net.predict(X_test)
    print(mean_squared_error(pred, y_test))
    print(mean_euclidean_error(pred, y_test))

    ml_cup_blind = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TS.csv', delimiter=','), 0, 1)
