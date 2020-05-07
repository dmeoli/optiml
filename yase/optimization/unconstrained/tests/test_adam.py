import numpy as np
import pytest

from yase.optimization.optimizer import quad1, quad2, Rosenbrock
from yase.optimization.unconstrained.stochastic import Adam


def test_Adam_quadratic():
    assert np.allclose(Adam(quad1, step_size=0.1).minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(Adam(quad2, step_size=0.1).minimize()[0], quad2.x_star(), rtol=0.1)


def test_Adam_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(Adam(rosen, step_size=0.1).minimize()[0], rosen.x_star(), rtol=0.1)


def test_Adam_standard_momentum_quadratic():
    assert np.allclose(Adam(quad1, momentum_type='standard').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(Adam(quad2, momentum_type='standard').minimize()[0], quad2.x_star(), rtol=0.1)


def test_Adam_standard_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(Adam(rosen, step_size=0.1, epochs=2000, momentum_type='standard').minimize()[0],
                       rosen.x_star(), rtol=0.1)


def test_Nadam_quadratic():
    assert np.allclose(Adam(quad1, momentum_type='nesterov').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(Adam(quad2, step_size=0.1, momentum_type='nesterov', momentum=0.5).minimize()[0],
                       quad2.x_star(), rtol=0.1)


def test_Nadam_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(Adam(rosen, momentum_type='nesterov').minimize()[0], rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
