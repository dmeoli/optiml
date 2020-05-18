import numpy as np
import pytest

from yase.optimization.box_constrained import ActiveSet, BoxConstrainedQuadratic


def test():
    assert np.allclose(ActiveSet(BoxConstrainedQuadratic()).minimize().x, 0.)


if __name__ == "__main__":
    pytest.main()
