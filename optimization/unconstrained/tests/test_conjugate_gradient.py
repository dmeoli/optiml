import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.conjugate_gradient import NonlinearConjugateGradient, QuadraticConjugateGradient


def test_QuadraticConjugateGradient():
    assert np.allclose(QuadraticConjugateGradient(quad1).minimize()[0], quad1.x_star())
    assert np.allclose(QuadraticConjugateGradient(quad2).minimize()[0], quad2.x_star())


def test_NonlinearConjugateGradient_quadratic_wf0():
    assert np.allclose(NonlinearConjugateGradient(quad1).minimize()[0], quad1.x_star())
    assert np.allclose(NonlinearConjugateGradient(quad2).minimize()[0], quad2.x_star())


def test_NonlinearConjugateGradient_Rosenbrock_wf0():
    obj = Rosenbrock()
    assert np.allclose(NonlinearConjugateGradient(obj, wf=0).minimize()[0], obj.x_star(), rtol=0.1)


def test_NonlinearConjugateGradient_quadratic_wf1():
    assert np.allclose(NonlinearConjugateGradient(quad1, wf=1).minimize()[0], quad1.x_star())
    assert np.allclose(NonlinearConjugateGradient(quad2, wf=1).minimize()[0], quad2.x_star())


def test_NonlinearConjugateGradient_Rosenbrock_wf1():
    obj = Rosenbrock()
    assert np.allclose(NonlinearConjugateGradient(obj, wf=1, max_iter=1500).minimize()[0], obj.x_star(), rtol=0.1)


def test_NonlinearConjugateGradient_quadratic_wf2():
    assert np.allclose(NonlinearConjugateGradient(quad1, wf=2).minimize()[0], quad1.x_star())
    assert np.allclose(NonlinearConjugateGradient(quad2, wf=2).minimize()[0], quad2.x_star())


def test_NonlinearConjugateGradient_Rosenbrock_wf2():
    obj = Rosenbrock()
    assert np.allclose(NonlinearConjugateGradient(obj, wf=2, max_iter=2000).minimize()[0], obj.x_star(), rtol=0.5)


def test_NonlinearConjugateGradient_quadratic_wf3():
    assert np.allclose(NonlinearConjugateGradient(quad1, wf=3).minimize()[0], quad1.x_star())
    assert np.allclose(NonlinearConjugateGradient(quad2, wf=3).minimize()[0], quad2.x_star())


def test_NonlinearConjugateGradient_Rosenbrock_wf3():
    obj = Rosenbrock()
    assert np.allclose(NonlinearConjugateGradient(obj, wf=3).minimize()[0], obj.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
