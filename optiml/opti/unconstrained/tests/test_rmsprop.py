import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.stochastic import RMSProp


def test_RMSProp_quadratic():
    assert np.allclose(RMSProp(f=quad1, step_size=0.1).minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(RMSProp(f=quad2, step_size=0.1).minimize().x, quad2.x_star(), rtol=0.1)


def test_RMSProp_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(RMSProp(f=rosen, epochs=1500).minimize().x, rosen.x_star(), rtol=0.1)


def test_RMSProp_Polyak_momentum_quadratic():
    assert np.allclose(RMSProp(f=quad1, momentum_type='polyak').minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(RMSProp(f=quad2, momentum_type='polyak').minimize().x, quad2.x_star(), rtol=0.1)


def test_RMSProp_Polyak_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(RMSProp(f=rosen, momentum_type='polyak').minimize().x, rosen.x_star(), rtol=0.1)


def test_Nadam_quadratic():
    assert np.allclose(RMSProp(f=quad1, momentum_type='nesterov').minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(RMSProp(f=quad2, momentum_type='nesterov').minimize().x, quad2.x_star(), rtol=0.1)


def test_Nadam_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(RMSProp(f=rosen, momentum_type='nesterov').minimize().x, rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
