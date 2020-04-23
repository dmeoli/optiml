import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

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
from optimization.unconstrained.quasi_newton import BFGS
from optimization.unconstrained.rmsprop import RMSProp
from optimization.unconstrained.rprop import RProp
from utils import load_ml_cup, mean_euclidean_error, load_ml_cup_blind

line_search_optimizers = [NonlinearConjugateGradient, SteepestDescentAcceleratedGradient,
                          SteepestGradientDescent, HeavyBallGradient, BFGS]

stochastic_adaptive_optimizers = [Adam, AdaMax, AMSGrad, AdaGrad, AdaDelta, RProp, RMSProp]

stochastic_optimizers = [GradientDescent, AcceleratedGradient]

if __name__ == '__main__':
    X, y = load_ml_cup()

    tuned_parameters = [{'layers': [(FullyConnected(20, 20, sigmoid),
                                     FullyConnected(20, 20, sigmoid),
                                     FullyConnected(20, 2, linear)),
                                    (FullyConnected(20, 20, tanh),
                                     FullyConnected(20, 20, tanh),
                                     FullyConnected(20, 2, linear))],
                         'optimizer': line_search_optimizers,
                         'epochs': [1000, 500, 200, 100],
                         'max_f_eval': [10000, 15000, 20000, 25000, 30000],
                         'batch_size': [300, 500, 700]},
                        {'layers': [(FullyConnected(20, 20, sigmoid),
                                     FullyConnected(20, 20, sigmoid),
                                     FullyConnected(20, 2, linear)),
                                    (FullyConnected(20, 20, tanh),
                                     FullyConnected(20, 20, tanh),
                                     FullyConnected(20, 2, linear))],
                         'optimizer': stochastic_optimizers + stochastic_adaptive_optimizers,
                         'learning_rate': [0.001, 0.01, 0.1],
                         'momentum_type': ['standard', 'nesterov'],
                         'momentum': [0.9, 0.8],
                         'epochs': [1000, 500, 200, 100],
                         'batch_size': [300, 500, 700]}]

    grid = GridSearchCV(NeuralNetworkRegressor(),
                        param_grid=tuned_parameters,
                        scoring=make_scorer(mean_euclidean_error, greater_is_better=False),
                        cv=5,  # 5 fold cross validation
                        n_jobs=-1,  # use all processors
                        refit=True,  # refit the best model on the full dataset
                        verbose=True)
    grid.fit(X, y)

    print(f'best parameters: {grid.best_params_}')
    print(f'best score: {-grid.best_score_}')

    # save predictions on the blind test set
    np.savetxt('./ml/data/ML-CUP19/dmeoli_ML-CUP19-TS_nn.csv', grid.predict(load_ml_cup_blind()), delimiter=',')
