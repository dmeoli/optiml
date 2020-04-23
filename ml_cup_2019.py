import numpy as np
from qpsolvers import solve_qp
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

from ml.svm import scipy_solve_qp
from optimization.constrained.active_set import ActiveSet
from optimization.constrained.frank_wolfe import FrankWolfe
from optimization.constrained.interior_point import InteriorPoint
from optimization.constrained.lagrangian_dual import LagrangianDual
from optimization.constrained.projected_gradient import ProjectedGradient
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
from utils import load_ml_cup, mean_euclidean_error, scipy_solve_svm, load_ml_cup_blind, plot_validation_curve, \
    plot_learning_curve

line_search_optimizers = [NonlinearConjugateGradient, SteepestGradientDescent, HeavyBallGradient, BFGS]

stochastic_optimizers = [GradientDescent, Adam, AdaMax, AMSGrad, AdaGrad, AdaDelta, RProp, RMSProp]

constrained_optimizers = [ProjectedGradient, ActiveSet, FrankWolfe, InteriorPoint, LagrangianDual, 'SMO']

other_constrained_optimizers = [solve_qp, scipy_solve_qp, scipy_solve_svm]

if __name__ == '__main__':
    X, y = load_ml_cup()

    gamma_range = [1e-4, 1e-2]
    C_range = [1, 10, 100, 1000]
    epsilon_range = [0.001, 0.01, 0.1, 0.2, 0.3]

    tuned_parameters = {'estimator__kernel': ['rbf'],
                        'estimator__epsilon': epsilon_range,
                        'estimator__C': C_range,
                        'estimator__gamma': gamma_range}

    from sklearn.svm import SVR as SKLSVR

    grid = GridSearchCV(MultiOutputRegressor(SKLSVR()), param_grid=tuned_parameters,
                        scoring=make_scorer(mean_euclidean_error, greater_is_better=False),
                        cv=5,  # 5 fold cross validation
                        n_jobs=-1,  # use all processors
                        refit=True,  # refit the best model on the full dataset
                        verbose=True)
    grid.fit(X, y)

    print(f'best parameters: {grid.best_params_}')
    print(f'best score: {-grid.best_score_}')

    scorer = make_scorer(mean_euclidean_error)

    # plot validation curve to visualize the performance metric over a
    # range of values for some hyperparameters (C, gamma, epsilon, etc.)
    for (param_name, param_range) in tuned_parameters.items():
        if 'kernel' not in param_name:
            plot_validation_curve(grid.best_estimator_, X, y, param_name, param_range, scorer)

    # plot learning curve to visualize the effect of the
    # number of observations on the performance metric
    plot_learning_curve(grid.best_estimator_, X, y, scorer)

    # save predictions on the blind test set
    np.savetxt('./ml/data/ML-CUP19/dmeoli_ML-CUP19-TS.csv', grid.predict(load_ml_cup_blind()), delimiter=',')
