import numpy as np
from qpsolvers import solve_qp
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

from ml.svm import SVR, scipy_solve_qp
from optimization.constrained.active_set import ActiveSet
from optimization.constrained.frank_wolfe import FrankWolfe
from optimization.constrained.interior_point import InteriorPoint
from optimization.constrained.lagrangian_dual import LagrangianDual
from optimization.constrained.projected_gradient import ProjectedGradient
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
from utils import load_ml_cup, load_ml_cup_blind, mean_euclidean_error, scipy_solve_svm

line_search_optimizers = [NonlinearConjugateGradient, SteepestDescentAcceleratedGradient,
                          SteepestGradientDescent, HeavyBallGradient, BFGS]

stochastic_adaptive_optimizers = [Adam, AdaMax, AMSGrad, AdaGrad, AdaDelta, RProp, RMSProp]

stochastic_optimizers = [GradientDescent, AcceleratedGradient]

constrained_optimizers = [ProjectedGradient, ActiveSet, FrankWolfe, InteriorPoint, LagrangianDual, 'SMO']

other_constrained_optimizers = [solve_qp, scipy_solve_qp, scipy_solve_svm]

if __name__ == '__main__':
    X, y = load_ml_cup()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    tuned_parameters = [{'estimator__kernel': ['linear'],
                         'estimator__C': [1, 10, 100],
                         'estimator__epsilon': [0.1, 0.01, 0.001]},
                        {'estimator__kernel': ['poly'],
                         'estimator__degree': [3., 4., 5.],
                         'estimator__gamma': ['auto', 'scale'],
                         'estimator__C': [1, 10, 100],
                         'estimator__epsilon': [0.1, 0.01, 0.001],
                         'estimator__coef0': [1, 10, 100, 1000]},
                        {'estimator__kernel': ['rbf', 'laplacian'],
                         'estimator__gamma': ['auto', 'scale'],
                         'estimator__C': [1, 10, 100],
                         'estimator__epsilon': [0.1, 0.01, 0.001]}]

    svr = GridSearchCV(MultiOutputRegressor(SVR()),
                       param_grid=tuned_parameters,
                       scoring=make_scorer(mean_euclidean_error, greater_is_better=False),
                       cv=None,  # if None changed from 3-fold to 5-fold
                       n_jobs=-1)  # use all processors
    svr.fit(X_train, y_train)

    print(f'best parameters set: {svr.best_params_}')
    print()
    print('grid scores:')
    print()
    means = svr.cv_results_['mean_test_score']
    stds = svr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svr.cv_results_['params']):
        print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
    print()
    print('mean euclidean error on the test set:')
    print()
    print(mean_euclidean_error(y_test, svr.predict(X_test)))

    X_blind_test = load_ml_cup_blind()

    # now retrain the best estimator on the full dataset
    svr = svr.best_estimator_
    svr.fit(X, y)
    # and then save predictions on the blind test set
    np.savetxt('./ml/data/ML-CUP19/dmeoli_ML-CUP19-TS_svr.csv', svr.predict(X_blind_test), delimiter=',')
