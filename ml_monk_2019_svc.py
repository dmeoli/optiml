from qpsolvers import solve_qp
from sklearn.svm import SVC as SKLSVC

from ml.metrics import accuracy_score
from ml.svm.kernels import *
from ml.svm.svm import SVC
from optimization.constrained.active_set import ActiveSet
from optimization.constrained.frank_wolfe import FrankWolfe
from optimization.constrained.interior_point import InteriorPoint
from optimization.constrained.lagrangian_dual import LagrangianDual
from optimization.constrained.projected_gradient import ProjectedGradient
from utils import load_monk

constrained_optimizers = [ProjectedGradient, ActiveSet, FrankWolfe, InteriorPoint, LagrangianDual, solve_qp]

if __name__ == '__main__':
    for i in (1, 2, 3):
        X_train, X_test, y_train, y_test = load_monk(i)
        svc = SVC(kernel=polynomial_kernel).fit(X_train, y_train, optimizer=ProjectedGradient, verbose=False)
        # print('w = ', svc.w)
        # print('b = ', svc.b)
        # print('idx of support vectors = ', svc.sv_idx)
        # print('support vectors = ', svc.sv)
        print("monk's #" + str(i) + " accuracy: " + str(accuracy_score(svc.predict(X_test), y_test)))
        # print()

        X_train, X_test, y_train, y_test = load_monk(i)
        svc = SKLSVC(kernel='poly').fit(X_train, y_train)
        # print('w = ', svc.coef_)
        # print('b = ', svc.intercept_)
        # print('idx of support vectors = ', svc.support_)
        # print('support vectors = ', svc.support_vectors_)
        print("sklearn monk's #" + str(i) + " accuracy: " + str(accuracy_score(svc.predict(X_test), y_test)))
        print()
