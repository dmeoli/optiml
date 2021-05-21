import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.stochastic import AMSGrad


def test_AMSGrad_quadratic():
    assert np.allclose(AMSGrad(f=quad1, step_size=0.1).minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(AMSGrad(f=quad2, step_size=0.1).minimize().x, quad2.x_star(), rtol=0.1)


def test_AMSGrad_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AMSGrad(f=rosen, step_size=0.1).minimize().x, rosen.x_star(), rtol=0.1)


def test_AMSGrad_Polyak_momentum_quadratic():
    assert np.allclose(AMSGrad(f=quad1, momentum_type='polyak').minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(AMSGrad(f=quad2, momentum_type='polyak').minimize().x, quad2.x_star(), rtol=0.1)


def test_AMSGrad_Polyak_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AMSGrad(f=rosen, momentum_type='polyak').minimize().x, rosen.x_star(), rtol=0.1)


def test_AMSGrad_nesterov_momentum_quadratic():
    assert np.allclose(AMSGrad(f=quad1, momentum_type='nesterov').minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(AMSGrad(f=quad2, momentum_type='nesterov').minimize().x, quad2.x_star(), rtol=0.1)


def test_AMSGrad_nesterov_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AMSGrad(f=rosen, momentum_type='nesterov').minimize().x, rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
