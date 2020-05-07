import numpy as np
import pytest

from yase.optimization.optimizer import quad1, Rosenbrock, quad2
from yase.optimization.unconstrained.stochastic import AMSGrad


def test_AMSGrad_quadratic():
    assert np.allclose(AMSGrad(quad1, step_size=0.1).minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AMSGrad(quad2, step_size=0.1).minimize()[0], quad2.x_star(), rtol=0.1)


def test_AMSGrad_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AMSGrad(rosen, step_size=0.1).minimize()[0], rosen.x_star(), rtol=0.1)


def test_AMSGrad_standard_momentum_quadratic():
    assert np.allclose(AMSGrad(quad1, momentum_type='standard').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AMSGrad(quad2, momentum_type='standard').minimize()[0], quad2.x_star(), rtol=0.1)


def test_AMSGrad_standard_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AMSGrad(rosen, momentum_type='standard').minimize()[0], rosen.x_star(), rtol=0.1)


def test_AMSGrad_nesterov_momentum_quadratic():
    assert np.allclose(AMSGrad(quad1, momentum_type='nesterov').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AMSGrad(quad2, momentum_type='nesterov').minimize()[0], quad2.x_star(), rtol=0.1)


def test_AMSGrad_nesterov_momentum_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AMSGrad(rosen, momentum_type='nesterov').minimize()[0], rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
