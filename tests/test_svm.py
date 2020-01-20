import numpy as np
import pytest

from ml.datasets import DataSet
from ml.learning import grade_learner, err_ratio
from ml.svm import MultiSVM


def test_svm():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    svm = MultiSVM()
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])
    svm.fit(X, y)
    assert svm.predict([[5.0, 3.1, 0.9, 0.1]]) == 0
    assert svm.predict([[5.1, 3.5, 1.0, 0.0]]) == 0
    assert svm.predict([[4.9, 3.3, 1.1, 0.1]]) == 0
    assert svm.predict([[6.0, 3.0, 4.0, 1.1]]) == 1
    assert svm.predict([[6.1, 2.2, 3.5, 1.0]]) == 1
    assert svm.predict([[5.9, 2.5, 3.3, 1.1]]) == 1
    assert svm.predict([[7.5, 4.1, 6.2, 2.3]]) == 2
    assert svm.predict([[7.3, 4.0, 6.1, 2.4]]) == 2
    assert svm.predict([[7.0, 3.3, 6.1, 2.5]]) == 2
    tests = [([5.0, 3.1, 0.9, 0.1], 0),
             ([5.1, 3.5, 1.0, 0.0], 0),
             ([4.9, 3.3, 1.1, 0.1], 0),
             ([6.0, 3.0, 4.0, 1.1], 1),
             ([6.1, 2.2, 3.5, 1.0], 1),
             ([5.9, 2.5, 3.3, 1.1], 1),
             ([7.5, 4.1, 6.2, 2.3], 2),
             ([7.3, 4.0, 6.1, 2.4], 2),
             ([7.0, 3.3, 6.1, 2.5], 2)]
    assert grade_learner(svm, tests) >= 1 / 3
    assert err_ratio(svm, iris) < 0.21


if __name__ == "__main__":
    pytest.main()
