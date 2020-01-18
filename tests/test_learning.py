import pytest

from ml.datasets import DataSet
from ml.learning import WeightedLearner, ada_boost, grade_learner, err_ratio
from ml.neural_network import PerceptronLearner


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
