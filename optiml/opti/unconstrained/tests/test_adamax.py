import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.stochastic import AdaMax


def test_AdaMax_quadratic():
    assert np.allclose(AdaMax(f=quad1, step_size=0.1).minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(f=quad2, step_size=0.1).minimize().x, quad2.x_star(), rtol=0.1)


def test_AdaMax_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaMax(f=rosen, step_size=0.1).minimize().x, rosen.x_star(), rtol=0.1)


def test_AdaMax_Polyak_momentum_quadratic():
    assert np.allclose(AdaMax(f=quad1, momentum_type='polyak').minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(f=quad2, momentum_type='polyak').minimize().x, quad2.x_star(), rtol=0.1)


def test_AdaMax_Polyak_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaMax(f=rosen, step_size=0.1, epochs=2000, momentum_type='polyak',
                              momentum=0.3).minimize().x, rosen.x_star(), rtol=0.1)


def test_NadaMax_quadratic():
    assert np.allclose(AdaMax(f=quad1, momentum_type='nesterov').minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(f=quad2, momentum_type='nesterov').minimize().x, quad2.x_star(), rtol=0.1)


def test_NadaMax_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaMax(f=rosen, momentum_type='nesterov', momentum=0.8).minimize().x, rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
