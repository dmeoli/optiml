import numpy as np
import pytest

from optiml.optimization.unconstrained import quad2, quad1, Rosenbrock
from optiml.optimization.unconstrained.stochastic import RProp


def test_RProp_quadratic():
    assert np.allclose(RProp(f=quad1, x=np.random.uniform(size=2)).minimize().x, quad1.x_star())
    assert np.allclose(RProp(f=quad2, x=np.random.uniform(size=2)).minimize().x, quad2.x_star())


def test_RProp_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(RProp(f=rosen, x=np.random.uniform(size=2)).minimize().x, rosen.x_star(), rtol=0.1)


def test_RProp_standard_momentum_quadratic():
    assert np.allclose(RProp(f=quad1, x=np.random.uniform(size=2), momentum_type='standard',
                             momentum=0.6).minimize().x, quad1.x_star())
    assert np.allclose(RProp(f=quad2, x=np.random.uniform(size=2), momentum_type='standard',
                             momentum=0.6).minimize().x, quad2.x_star())


def test_RProp_standard_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(RProp(f=rosen, x=np.random.uniform(size=2), momentum_type='standard',
                             momentum=0.6).minimize().x, rosen.x_star(), rtol=0.1)


def test_RProp_nesterov_momentum_quadratic():
    assert np.allclose(RProp(f=quad1, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       quad1.x_star())
    assert np.allclose(RProp(f=quad2, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       quad2.x_star())


def test_RProp_nesterov_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(RProp(f=rosen, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
