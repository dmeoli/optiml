import numpy as np
import pytest

from optimization import gen_quad_1, gen_quad_2, gen_quad_5, Rosenbrock
from optimization.unconstrained.proximal_bundle import ProximalBundle


def test_quadratic():
    x, _ = ProximalBundle(gen_quad_1)
    assert np.allclose(gen_quad_1.jacobian(x), 0)

    x, _ = ProximalBundle(gen_quad_2)
    assert np.allclose(gen_quad_2.jacobian(x), 0)

    # x, _ = ProximalBundle(gen_quad_3)
    # assert np.allclose(gen_quad_3.jacobian(x), 0)

    # x, _ = ProximalBundle(gen_quad_4)
    # assert np.allclose(gen_quad_4.jacobian(x), 0)

    x, _ = ProximalBundle(gen_quad_5)
    assert np.allclose(gen_quad_5.jacobian(x), 0)


def test_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = ProximalBundle(obj)
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = ProximalBundle(obj)
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
