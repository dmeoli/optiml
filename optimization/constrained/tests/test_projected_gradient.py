import numpy as np
import pytest

from optimization.constrained.projected_gradient import ProjectedGradient
from optimization.optimization_function import BoxConstrained


def test():
    np.random.seed(2)
    assert np.allclose(ProjectedGradient(BoxConstrained(ndim=2)).minimize()[0], 0.)


if __name__ == "__main__":
    pytest.main()
