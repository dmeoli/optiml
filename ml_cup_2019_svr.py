import numpy as np
from qpsolvers import solve_qp
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR as SKLSVR

from ml.learning import MultiTargetRegressor
from ml.losses import mean_squared_error
from ml.metrics import mean_euclidean_error
from ml.svm.kernels import linear_kernel, polynomial_kernel, rbf_kernel
from ml.svm.svm import SVR, scipy_solve_qp
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
from utils import load_ml_cup

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    svr = MultiTargetRegressor(SVR(kernel=rbf_kernel, eps=0.1))
    svr.fit(X_train, y_train, optimizer=ProjectedGradient, verbose=False)
    pred = svr.predict(X_test)
    print('mse: ', mean_squared_error(pred, y_test))
    print('mee: ', mean_euclidean_error(pred, y_test))
    print()

    svr = MultiOutputRegressor(SKLSVR(kernel='rbf', epsilon=0.1)).fit(X_train, y_train)
    pred = svr.predict(X_test)
    print('sklearn mse: ', mean_squared_error(pred, y_test))
    print('sklearn mee: ', mean_euclidean_error(pred, y_test))
