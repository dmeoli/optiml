from qpsolvers import solve_qp
from sklearn.svm import SVC as SKLSVC

from ml.metrics import accuracy_score
from ml.kernels import linear_kernel, polynomial_kernel, rbf_kernel
from ml.svm import SVC, scipy_solve_qp
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
from utils import load_monk

line_search_optimizers = [NonlinearConjugateGradient, SteepestDescentAcceleratedGradient,
                          SteepestGradientDescent, HeavyBallGradient, BFGS]

stochastic_adaptive_optimizers = [Adam, AdaMax, AMSGrad, AdaGrad, AdaDelta, RProp, RMSProp]

stochastic_optimizers = [GradientDescent, AcceleratedGradient]

others = [ProximalBundle]

constrained_optimizers = [ProjectedGradient, ActiveSet, FrankWolfe, InteriorPoint,
                          LagrangianDual, solve_qp, scipy_solve_qp]

kernels = [linear_kernel, polynomial_kernel, rbf_kernel]

if __name__ == '__main__':
    for i in (1, 2, 3):
        X_train, X_test, y_train, y_test = load_monk(i)
        svc = SVC(kernel=polynomial_kernel, optimizer=ProjectedGradient, verbose=False).fit(X_train, y_train)
        print("monk #" + str(i) + " accuracy: " + str(accuracy_score(svc.predict(X_test), y_test)))

        X_train, X_test, y_train, y_test = load_monk(i)
        sklsvc = SKLSVC(kernel='poly').fit(X_train, y_train)
        print("sklearn monk #" + str(i) + " accuracy: " + str(accuracy_score(sklsvc.predict(X_test), y_test)))
        print()
