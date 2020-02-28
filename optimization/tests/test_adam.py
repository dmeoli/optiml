import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.adam import Adam


def test_Adam_standard_momentum_quadratic():
    assert np.allclose(Adam(quad1, momentum_type='standard').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(Adam(quad2, momentum_type='standard').minimize()[0], quad2.x_star(), rtol=0.01)


def test_Adam_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(Adam(obj, momentum_type='standard', max_iter=2000).minimize()[0], obj.x_star(), rtol=0.1)


def test_Nadam_quadratic():
    assert np.allclose(Adam(quad1, momentum_type='nesterov').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(Adam(quad2, momentum_type='nesterov').minimize()[0], quad2.x_star(), rtol=0.01)


def test_Nadam_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(Adam(obj, momentum_type='nesterov').minimize()[0], obj.x_star(), rtol=0.15)


if __name__ == "__main__":
    pytest.main()
