import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.conjugate_gradient import NonlinearConjugateGradient, ConjugateGradientQuadratic


def test_ConjugateGradientQuadratic():
    x, _ = ConjugateGradientQuadratic(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = ConjugateGradientQuadratic(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_NonlinearConjugateGradient_quadratic_wf0():
    x, _ = NonlinearConjugateGradient(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = NonlinearConjugateGradient(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_NonlinearConjugateGradient_quadratic_wf1():
    x, _ = NonlinearConjugateGradient(quad1, wf=1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = NonlinearConjugateGradient(quad2, wf=1).minimize()
    assert np.allclose(x, quad2.x_star())


def test_NonlinearConjugateGradient_quadratic_wf2():
    x, _ = NonlinearConjugateGradient(quad1, wf=2).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = NonlinearConjugateGradient(quad2, wf=2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_NonlinearConjugateGradient_quadratic_wf3():
    x, _ = NonlinearConjugateGradient(quad1, wf=3).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = NonlinearConjugateGradient(quad2, wf=3).minimize()
    assert np.allclose(x, quad2.x_star())


def test_NonlinearConjugateGradient_Rosenbrock():
    obj = Rosenbrock()
    x, _ = NonlinearConjugateGradient(obj, wf=3).minimize()
    assert np.allclose(x, obj.x_star())


if __name__ == "__main__":
    pytest.main()
