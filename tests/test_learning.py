import pytest

from ml.datasets import DataSet, open_data, parse_csv
from ml.neural_network import PerceptronLearner
from ml.learning import *
from ml.svm import MultiSVM


def test_exclude():
    iris = DataSet(name='iris', exclude=[3])
    assert iris.inputs == [0, 1, 2]


def test_parse_csv():
    iris = open_data('iris.csv').read()
    assert parse_csv(iris)[0] == [5.1, 3.5, 1.4, 0.2, 'setosa']


def test_weighted_mode():
    assert weighted_mode('abbaa', [1, 2, 3, 1, 2]) == 'b'


def test_weighted_replicate():
    assert weighted_replicate('ABC', [1, 2, 1], 4) == ['A', 'B', 'B', 'C']


def test_means_and_deviation():
    iris = DataSet(name='iris')
    means, deviations = iris.find_means_and_deviations()
    assert round(means['setosa'][0], 3) == 5.006
    assert round(means['versicolor'][0], 3) == 5.936
    assert round(means['virginica'][0], 3) == 6.588
    assert round(deviations['setosa'][0], 3) == 0.352
    assert round(deviations['versicolor'][0], 3) == 0.516
    assert round(deviations['virginica'][0], 3) == 0.636


def test_k_nearest_neighbors():
    iris = DataSet(name='iris')
    knn = NearestNeighborLearner(iris, k=3)
    assert knn([5, 3, 1, 0.1]) == 'setosa'
    assert knn([6, 5, 3, 1.5]) == 'versicolor'
    assert knn([7.5, 4, 6, 2]) == 'virginica'


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


def test_random_weights():
    min_value = -0.5
    max_value = 0.5
    num_weights = 10
    test_weights = random_weights(min_value, max_value, num_weights)
    assert len(test_weights) == num_weights
    for weight in test_weights:
        assert min_value <= weight <= max_value


def test_ada_boost():
    iris = DataSet(name='iris')
    iris.classes_to_numbers()
    wl = WeightedLearner(PerceptronLearner)
    ab = ada_boost(iris, wl, 5)
    tests = [([5, 3, 1, 0.1], 0),
             ([5, 3.5, 1, 0], 0),
             ([6, 3, 4, 1.1], 1),
             ([6, 2, 3.5, 1], 1),
             ([7.5, 4, 6, 2], 2),
             ([7, 3, 6, 2.5], 2)]
    assert grade_learner(ab, tests) > 4 / 6
    assert err_ratio(ab, iris) < 0.25


if __name__ == "__main__":
    pytest.main()
