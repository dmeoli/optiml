import numpy as np
import pytest

from optiml.optimization import quad1, quad2
from optiml.optimization.unconstrained import Rosenbrock
from optiml.optimization.unconstrained.stochastic import AdaMax


def test_AdaMax_quadratic():
    assert np.allclose(AdaMax(f=quad1, x=np.random.uniform(size=2), step_size=0.1).minimize().x,
                       quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(f=quad2, x=np.random.uniform(size=2), step_size=0.1).minimize().x,
                       quad2.x_star(), rtol=0.1)


def test_AdaMax_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaMax(f=rosen, x=np.random.uniform(size=2), step_size=0.1).minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_AdaMax_standard_momentum_quadratic():
    assert np.allclose(AdaMax(f=quad1, x=np.random.uniform(size=2), momentum_type='standard').minimize().x,
                       quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(f=quad2, x=np.random.uniform(size=2), momentum_type='standard').minimize().x,
                       quad2.x_star(), rtol=0.1)


def test_AdaMax_standard_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaMax(f=rosen, x=np.random.uniform(size=2), step_size=0.1, epochs=2000,
                              momentum_type='standard', momentum=0.3).minimize().x, rosen.x_star(), rtol=0.1)


def test_NadaMax_quadratic():
    assert np.allclose(AdaMax(f=quad1, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaMax(f=quad2, x=np.random.uniform(size=2), momentum_type='nesterov').minimize().x,
                       quad2.x_star(), rtol=0.1)


def test_NadaMax_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaMax(f=rosen, x=np.random.uniform(size=2), momentum_type='nesterov',
                              momentum=0.8).minimize().x, rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
