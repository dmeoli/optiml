import numpy as np

from ml.losses import mean_squared_error
from ml.metrics import mean_euclidean_error
from ml.neural_network.activations import sigmoid, tanh, relu, linear
from ml.neural_network.layers import Dense
from ml.neural_network.neural_network import NeuralNetwork
from ml.regularizers import l1
from optimization.unconstrained.accelerated_gradient import AcceleratedGradient
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

stochastic_optimizers = [Adam, AMSGrad, AdaGrad, AdaDelta, AdaMax, GradientDescent, RMSProp, RProp]

line_search_optimizers = [NonlinearConjugateGradient, AcceleratedGradient, Newton,
                          SteepestGradientDescent, HeavyBallGradient, BFGS]

activations = [sigmoid, tanh, relu]

losses = [mean_squared_error]

if __name__ == '__main__':
    # X, y = load_iris(return_X_y=True)

    # grid = []
    # for e in itertools.product(*args):
    #     grid.append({'learning_rate': e[0],
    #                  'epochs': e[1],
    #                  'momentum': e[2],
    #                  'lmbda': e[3],
    #                  'n_hidden': e[4],
    #                  'batch_size': e[5],
    #                  'n_folds': e[6],
    #                  'activation': e[7]})

    # net = NeuralNetwork(Dense(4, 4, sigmoid),
    #                     Dense(4, 4, sigmoid),
    #                     Dense(4, 3, softmax))
    #
    # net.fit(X, y, loss=cross_entropy, optimizer=GradientDescent, learning_rate=0.01, momentum_type='none',
    #         momentum=0.9, regularizer=l2, lmbda=0., epochs=300, batch_size=None, verbose=True, plot=True)
    # pred = net.predict(X)
    # print(pred)
    # print(accuracy_score(pred, y))

    ml_cup = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X, y = ml_cup[:, :-2], ml_cup[:, -2:]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    net = NeuralNetwork(Dense(20, 20, sigmoid),
                        Dense(20, 20, sigmoid),
                        Dense(20, 2, linear))

    net.fit(X_train, y_train, loss=mean_squared_error, optimizer=Adam, learning_rate=0.01,
            momentum_type='none', momentum=0.7, regularizer=l1, lmbda=0., epochs=1000, batch_size=None,
            verbose=True, plot=True)
    pred = net.predict(X_test)
    print(mean_squared_error(pred, y_test))
    print(mean_euclidean_error(pred, y_test))
