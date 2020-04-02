import numpy as np
from qpsolvers import solve_qp
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

from ml.kernels import linear_kernel, polynomial_kernel, rbf_kernel
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
from optimization.unconstrained.proximal_bundle import ProximalBundle
from optimization.unconstrained.quasi_newton import BFGS
from optimization.unconstrained.rmsprop import RMSProp
from optimization.unconstrained.rprop import RProp
from utils import load_ml_cup, load_ml_cup_blind, mean_euclidean_error

line_search_optimizers = [NonlinearConjugateGradient, SteepestDescentAcceleratedGradient,
                          SteepestGradientDescent, HeavyBallGradient, BFGS]

stochastic_adaptive_optimizers = [Adam, AdaMax, AMSGrad, AdaGrad, AdaDelta, RProp, RMSProp]

stochastic_optimizers = [GradientDescent, AcceleratedGradient]

others = [ProximalBundle]

constrained_optimizers = [ProjectedGradient, ActiveSet, FrankWolfe, InteriorPoint,
                          LagrangianDual, solve_qp, scipy_solve_qp]

kernels = [linear_kernel, polynomial_kernel, rbf_kernel]

if __name__ == '__main__':
    X, y = load_ml_cup()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    tuned_parameters = [{'estimator__kernel': [linear_kernel, polynomial_kernel, rbf_kernel],
                         'estimator__optimizer': [ProjectedGradient, FrankWolfe, InteriorPoint],
                         'estimator__gamma': ['auto', 'scale']}]

    svr = GridSearchCV(MultiOutputRegressor(SVR()),
                       param_grid=tuned_parameters,
                       scoring=make_scorer(mean_euclidean_error, greater_is_better=False),
                       cv=5).fit(X_train, y_train)

    print('best parameters set found on development set:')
    print()
    print(svr.best_params_)
    print()
    print('grid scores on development set:')
    print()
    means = svr.cv_results_['mean_test_score']
    stds = svr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svr.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    # print(svr.cv_results_)
    # print()
    print('the model is trained on the full development set.')
    print('the scores are computed on the full evaluation set.')
    print()
    print(mean_euclidean_error(y_test, svr.predict(X_test)))

    X_blind_test = load_ml_cup_blind()
    svr.fit(X, y)
    np.savetxt('./ml/data/ML-CUP19/dmeoli-svr.csv', svr.predict(X_blind_test), delimiter=',')
