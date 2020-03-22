from qpsolvers import solve_qp
from sklearn.model_selection import train_test_split

from ml.learning import MultiTargetRegressor
from ml.losses import mean_squared_error
from ml.metrics import mean_euclidean_error
from ml.svm.kernels import *
from ml.svm.svm import SVR, scipy_solve_qp
from optimization.constrained.active_set import ActiveSet
from optimization.constrained.frank_wolfe import FrankWolfe
from optimization.constrained.interior_point import InteriorPoint
from optimization.constrained.lagrangian_dual import LagrangianDual
from optimization.constrained.projected_gradient import ProjectedGradient

constrained_optimizers = [ProjectedGradient, ActiveSet, FrankWolfe, InteriorPoint,
                          LagrangianDual, solve_qp, scipy_solve_qp]

if __name__ == '__main__':
    ml_cup_train = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X, y = ml_cup_train[:, :-2], ml_cup_train[:, -2:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

    svr = MultiTargetRegressor(SVR(kernel=rbf_kernel, eps=0.1))
    svr.fit(X_train, y_train, optimizer=scipy_solve_qp, verbose=False)
    # for learner in svr.learners:
    #     print('w = ', learner.w)
    #     print('b = ', learner.b)
    #     print('idx of support vectors = ', learner.sv_idx)
    #     print('support vectors = ', learner.sv)
    pred = svr.predict(X_test)
    print('mse: ', mean_squared_error(pred, y_test))
    print('mee: ', mean_euclidean_error(pred, y_test))
    print()

    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import SVR

    svr = MultiOutputRegressor(SVR(kernel='rbf', epsilon=0.1)).fit(X_train, y_train)
    # for learner in svr.estimators_:
    #     print('w = ', learner.coef_)
    #     print('b = ', learner.intercept_)
    #     print('idx of support vectors = ', learner.support_)
    #     print('support vectors = ', learner.support_vectors_)
    pred = svr.predict(X_test)
    print('sklearn mse: ', mean_squared_error(pred, y_test))
    print('sklearn mee: ', mean_euclidean_error(pred, y_test))
