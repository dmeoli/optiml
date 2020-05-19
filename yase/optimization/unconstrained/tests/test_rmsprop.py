import numpy as np
import pytest

from yase.optimization.unconstrained import quad2, quad1, Rosenbrock
from yase.optimization.unconstrained.stochastic import RMSProp


def test_RMSProp_quadratic():
    assert np.allclose(RMSProp(f=quad1, x=np.random.uniform(size=2), step_size=0.1).minimize().x,
                       quad1.x_star(), rtol=0.1)
    assert np.allclose(RMSProp(f=quad2, x=np.random.uniform(size=2), step_size=0.1).minimize().x,
                       quad2.x_star(), rtol=0.1)


def test_RMSProp_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(RMSProp(f=rosen, x=np.random.uniform(size=2), epochs=1500).minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_RMSProp_standard_momentum_quadratic():
    assert np.allclose(RMSProp(f=quad1, x=np.random.uniform(size=2), momentum_type='standard').minimize().x,
                       quad1.x_star(), rtol=0.1)
    assert np.allclose(RMSProp(f=quad2, x=np.random.uniform(size=2), momentum_type='standard').minimize().x,
                       quad2.x_star(), rtol=0.1)


def test_RMSProp_standard_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(RMSProp(f=rosen, x=np.random.uniform(size=2), momentum_type='standard').minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_RMSProp_nesterov_momentum_quadratic():
    assert np.allclose(RMSProp(f=quad1, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       quad1.x_star(), rtol=0.1)
    assert np.allclose(RMSProp(f=quad2, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       quad2.x_star(), rtol=0.1)


def test_RMSProp_nesterov_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(RMSProp(f=rosen, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
