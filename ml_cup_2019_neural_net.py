import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV

from ml.losses import mean_squared_error
from ml.activations import sigmoid, tanh, linear
from ml.layers import FullyConnected
from ml.neural_network import NeuralNetworkRegressor
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
from utils import load_ml_cup, mean_euclidean_error, load_ml_cup_blind

line_search_optimizers = [NonlinearConjugateGradient, SteepestDescentAcceleratedGradient,
                          SteepestGradientDescent, HeavyBallGradient, BFGS]

stochastic_adaptive_optimizers = [Adam, AdaMax, AMSGrad, AdaGrad, AdaDelta, RProp, RMSProp]

stochastic_optimizers = [GradientDescent, AcceleratedGradient]

other_optimizers = [ProximalBundle]

if __name__ == '__main__':
    X, y = load_ml_cup()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    tuned_parameters = [{'layers': [(FullyConnected(20, 20, sigmoid),
                                     FullyConnected(20, 20, sigmoid),
                                     FullyConnected(20, 2, linear)),
                                    (FullyConnected(20, 20, tanh),
                                     FullyConnected(20, 20, tanh),
                                     FullyConnected(20, 2, linear))],
                         'loss': [mean_squared_error],
                         'optimizer': line_search_optimizers,
                         'epochs': [1000, 500, 200, 100],
                         'max_f_eval': [10000, 15000, 20000, 25000, 30000],
                         'batch_size': [300, 500, 700]},
                        {'layers': [(FullyConnected(20, 20, sigmoid),
                                     FullyConnected(20, 20, sigmoid),
                                     FullyConnected(20, 2, linear)),
                                    (FullyConnected(20, 20, tanh),
                                     FullyConnected(20, 20, tanh),
                                     FullyConnected(20, 2, tanh))],
                         'loss': [mean_squared_error],
                         'optimizer': stochastic_optimizers + stochastic_adaptive_optimizers + other_optimizers,
                         'learning_rate': [0.001, 0.01, 0.1],
                         'momentum_type': ['standard', 'nesterov'],
                         'momentum': [0.9, 0.8],
                         'epochs': [1000, 500, 200, 100],
                         'batch_size': [300, 500, 700]}]

    net = GridSearchCV(NeuralNetworkRegressor(),
                       param_grid=tuned_parameters,
                       scoring=make_scorer(mean_euclidean_error, greater_is_better=False),
                       cv=5)
    net.fit(X_train, y_train)

    print('best parameters set found on development set:')
    print()
    print(net.best_params_)
    print()
    print('grid scores on development set:')
    print()
    means = net.cv_results_['mean_test_score']
    stds = net.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, net.cv_results_['params']):
        print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
    print()
    print('the model is trained on the full development set.')
    print('the scores are computed on the full evaluation set.')
    print()
    print('mean euclidean error on the tes set:')
    print()
    print(mean_euclidean_error(y_test, net.predict(X_test)))

    X_blind_test = load_ml_cup_blind()

    # now retrain the best model on the full dataset
    net.fit(X, y)
    # and then save predictions on the blind test set
    np.savetxt('./ml/data/ML-CUP19/dmeoli_ML-CUP19-TS_nn.csv', net.predict(X_blind_test), delimiter=',')
