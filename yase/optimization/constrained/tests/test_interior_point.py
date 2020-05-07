import numpy as np
import pytest

from yase.optimization.constrained import BoxConstrainedQuadratic, InteriorPoint


def test():
    np.random.seed(2)
    assert np.allclose(InteriorPoint(BoxConstrainedQuadratic(ndim=2)).minimize()[0], 0.)


if __name__ == "__main__":
    pytest.main()
