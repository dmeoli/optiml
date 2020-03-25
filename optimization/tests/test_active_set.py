import numpy as np
import pytest

from optimization.constrained.active_set import ActiveSet
from optimization.optimization_function import BoxConstrainedQuadratic


def test():
    np.random.seed(0)
    assert np.isclose(ActiveSet(BoxConstrainedQuadratic()).minimize()[1], -2069.61640)


if __name__ == "__main__":
    pytest.main()
