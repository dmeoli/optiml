import numpy as np
import pytest

from yase.optimization.optimizer import quad1, quad2, Rosenbrock
from yase.optimization.unconstrained.stochastic import AdaMax


def test_AdaMax_quadratic():
    assert np.allclose(AdaMax(quad1, step_size=0.1).minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(quad2, step_size=0.1).minimize()[0], quad2.x_star(), rtol=0.1)


def test_AdaMax_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaMax(rosen, step_size=0.1).minimize()[0], rosen.x_star(), rtol=0.1)


def test_AdaMax_standard_momentum_quadratic():
    assert np.allclose(AdaMax(quad1, momentum_type='standard').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(quad2, momentum_type='standard').minimize()[0], quad2.x_star(), rtol=0.1)


def test_AdaMax_standard_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaMax(rosen, step_size=0.1, epochs=2000, momentum_type='standard', momentum=0.3).minimize()[0],
                       rosen.x_star(), rtol=0.1)


def test_NadaMax_quadratic():
    assert np.allclose(AdaMax(quad1, momentum_type='nesterov').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(quad2, momentum_type='nesterov').minimize()[0], quad2.x_star(), rtol=0.1)


def test_NadaMax_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaMax(rosen, momentum_type='nesterov', momentum=0.8).minimize()[0], rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
