import numpy as np
import pytest

from optiml.optimization.unconstrained import quad1, quad2, Rosenbrock
from optiml.optimization.unconstrained.stochastic import AdaDelta


def test_AdaDelta_quadratic():
    assert np.allclose(AdaDelta(f=quad1, x=np.random.uniform(size=2)).minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaDelta(f=quad2, x=np.random.uniform(size=2)).minimize().x, quad2.x_star(), rtol=0.1)


def test_AdaDelta_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaDelta(f=rosen, x=np.random.uniform(size=2), step_size=0.1).minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_AdaDelta_standard_momentum_quadratic():
    assert np.allclose(AdaDelta(f=quad1, x=np.random.uniform(size=2), momentum_type='standard').minimize().x,
                       quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaDelta(f=quad2, x=np.random.uniform(size=2), momentum_type='standard').minimize().x,
                       quad2.x_star(), rtol=0.1)


def test_AdaDelta_standard_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaDelta(f=rosen, x=np.random.uniform(size=2), momentum_type='standard').minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_AdaDelta_nesterov_momentum_quadratic():
    assert np.allclose(AdaDelta(f=quad1, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaDelta(f=quad2, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       quad2.x_star(), rtol=0.1)


def test_AdaDelta_nesterov_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaDelta(f=rosen, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       rosen.x_star(), rtol=0.01)


if __name__ == "__main__":
    pytest.main()
