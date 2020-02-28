import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.adamax import AdaMax


def test_AdaMax_standard_momentum_quadratic():
    assert np.allclose(AdaMax(quad1, momentum_type='standard').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(quad2, momentum_type='standard').minimize()[0], quad2.x_star(), rtol=0.1)


def test_AdaMax_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(AdaMax(obj, momentum_type='standard').minimize()[0], obj.x_star(), rtol=0.2)


def test_AdaMax_nesterov_momentum_quadratic():
    assert np.allclose(AdaMax(quad1, momentum_type='nesterov').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(quad2, momentum_type='nesterov').minimize()[0], quad2.x_star(), rtol=0.1)


def test_AdaMax_nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(AdaMax(obj, momentum_type='nesterov').minimize()[0], obj.x_star(), rtol=0.2)


if __name__ == "__main__":
    pytest.main()
