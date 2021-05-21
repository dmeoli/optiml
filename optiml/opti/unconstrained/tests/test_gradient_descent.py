import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.line_search import SteepestGradientDescent
from optiml.opti.unconstrained.stochastic import StochasticGradientDescent


def test_SteepestGradientDescent_quadratic():
    assert np.allclose(SteepestGradientDescent(f=quad1).minimize().x, quad1.x_star())
    assert np.allclose(SteepestGradientDescent(f=quad2).minimize().x, quad2.x_star())


def test_SteepestGradientDescent_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(SteepestGradientDescent(f=rosen).minimize().x, rosen.x_star())


def test_StochasticGradientDescent_quadratic():
    assert np.allclose(StochasticGradientDescent(f=quad1).minimize().x, quad1.x_star())
    assert np.allclose(StochasticGradientDescent(f=quad2).minimize().x, quad2.x_star())


def test_StochasticGradientDescent_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(StochasticGradientDescent(f=rosen).minimize().x, rosen.x_star(), rtol=0.1)


def test_StochasticGradientDescent_Polyak_momentum_quadratic():
    assert np.allclose(StochasticGradientDescent(f=quad1, momentum_type='polyak').minimize().x, quad1.x_star())
    assert np.allclose(StochasticGradientDescent(f=quad2, momentum_type='polyak').minimize().x, quad2.x_star())


def test_StochasticGradientDescent_Polyak_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(StochasticGradientDescent(f=rosen, momentum_type='polyak').minimize().x, rosen.x_star())


def test_StochasticGradientDescent_Nesterov_momentum_quadratic():
    assert np.allclose(StochasticGradientDescent(f=quad1, momentum_type='nesterov').minimize().x, quad1.x_star())
    assert np.allclose(StochasticGradientDescent(f=quad2, momentum_type='nesterov').minimize().x, quad2.x_star())


def test_StochasticGradientDescent_Nesterov_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(StochasticGradientDescent(f=rosen, momentum_type='nesterov').minimize().x, rosen.x_star())


if __name__ == "__main__":
    pytest.main()
