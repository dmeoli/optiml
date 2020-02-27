from sklearn.datasets import load_iris

from ml.losses import cross_entropy
from ml.metrics import accuracy_score
from ml.neural_network.activations import sigmoid, softmax
from ml.neural_network.layers import Dense
from ml.neural_network.neural_network import NeuralNetwork
from ml.regularizers import l2
from optimization.unconstrained.adadelta import AdaDelta
from optimization.unconstrained.adagrad import AdaGrad
from optimization.unconstrained.adam import Adam
from optimization.unconstrained.amsgrad import AMSGrad
from optimization.unconstrained.conjugate_gradient import NonlinearConjugateGradient
from optimization.unconstrained.gradient_descent import GradientDescent, SteepestGradientDescent
from optimization.unconstrained.heavy_ball_gradient import HeavyBallGradient
from optimization.unconstrained.newton import Newton
from optimization.unconstrained.quasi_newton import BFGS
from optimization.unconstrained.rmsprop import RMSProp
from optimization.unconstrained.rprop import RProp
from optimization.unconstrained.subgradient import Subgradient

optimizers = [Adam, AMSGrad, AdaGrad, AdaDelta, NonlinearConjugateGradient, GradientDescent,
              SteepestGradientDescent, HeavyBallGradient, BFGS, RMSProp, RProp]

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)

    net = NeuralNetwork(Dense(4, 4, sigmoid),
                        Dense(4, 4, sigmoid),
                        Dense(4, 3, softmax))

    net.fit(X, y, loss=cross_entropy, optimizer=Newton, regularizer=l2, lmbda=0.01,
            epochs=1000, batch_size=None, verbose=True)
    pred = net.predict(X)
    print(pred)
    print(accuracy_score(pred, y))

    # ml_cup = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    # X, y = ml_cup[:, :-2], ml_cup[:, -2:]
    #
    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    #
    # net = NeuralNetwork(Dense(20, 20, sigmoid),
    #                     Dense(20, 20, sigmoid),
    #                     Dense(20, 2, linear))
    #
    # net.fit(X_train, y_train, loss=mean_squared_error, optimizer=Adam, learning_rate=0.01,
    #         epochs=1000, batch_size=None, verbose=True)
    # pred = net.predict(X_test)
    # print(mean_squared_error(pred, y_test))
    # print(mean_euclidean_error(pred, y_test))
