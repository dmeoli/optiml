import numpy as np
from qpsolvers import solve_qp
from sklearn.model_selection import train_test_split

from ml.learning import MultiOutputLearner
from ml.losses import mean_squared_error
from ml.metrics import mean_euclidean_error
from ml.svm.kernels import rbf_kernel
from ml.svm.svm import SVR, scipy_solve_qp
from optimization.constrained.active_set import ActiveSet
from optimization.constrained.frank_wolfe import FrankWolfe
from optimization.constrained.interior_point import InteriorPoint
from optimization.constrained.lagrangian_dual import LagrangianDual
from optimization.constrained.projected_gradient import ProjectedGradient

constrained_optimizers = [ProjectedGradient, ActiveSet, FrankWolfe, InteriorPoint, LagrangianDual,
                          solve_qp, scipy_solve_qp]

if __name__ == '__main__':
    ml_cup_train = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X, y = ml_cup_train[:, :-2], ml_cup_train[:, -2:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    svr = MultiOutputLearner(SVR(kernel=rbf_kernel, eps=0.1))
    svr.fit(X_train, y_train, optimizer=ProjectedGradient, verbose=False)
    pred = svr.predict(X_test)
    print(mean_squared_error(pred, y_test))
    print(mean_euclidean_error(pred, y_test))

    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import SVR

    svr_sk = MultiOutputRegressor(SVR(kernel='rbf', epsilon=0.1)).fit(X_train, y_train)
    pred = svr_sk.predict(X_test)
    print(mean_squared_error(pred, y_test))
    print(mean_euclidean_error(pred, y_test))
