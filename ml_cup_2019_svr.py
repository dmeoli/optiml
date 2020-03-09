import numpy as np
from sklearn.model_selection import train_test_split

from ml.kernels import rbf_kernel
from ml.learning import MultiOutputLearner
from ml.losses import mean_squared_error
from ml.metrics import mean_euclidean_error
from ml.svm import SVR

if __name__ == '__main__':
    ml_cup_train = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X, y = ml_cup_train[:, :-2], ml_cup_train[:, -2:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    svr = MultiOutputLearner(SVR(kernel=rbf_kernel, degree=3., eps=0.1)).fit(X_train, y_train)
    pred = svr.predict(X_test)
    print(mean_squared_error(pred, y_test))
    print(mean_euclidean_error(pred, y_test))

    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import SVR

    svr_sk = MultiOutputRegressor(SVR(kernel='rbf', degree=3., epsilon=0.1)).fit(X_train, y_train)
    pred = svr_sk.predict(X_test)
    print(mean_squared_error(pred, y_test))
    print(mean_euclidean_error(pred, y_test))
