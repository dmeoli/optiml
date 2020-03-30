import numpy as np
import pytest

from optimization.constrained.active_set import ActiveSet
from optimization.optimization_function import BoxConstrainedQuadratic


def test():
    np.random.seed(2)
    assert np.allclose(ActiveSet(BoxConstrainedQuadratic(n=2)).minimize()[0], 0.)


if __name__ == "__main__":
    pytest.main()
