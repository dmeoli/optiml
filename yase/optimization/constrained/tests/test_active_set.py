import numpy as np
import pytest

from yase.optimization.constrained import ActiveSet, BoxConstrainedQuadratic


def test():
    np.random.seed(2)
    assert np.allclose(ActiveSet(BoxConstrainedQuadratic(ndim=2)).minimize().x, 0.)


if __name__ == "__main__":
    pytest.main()
