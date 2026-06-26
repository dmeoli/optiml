import numpy as np
import pytest

from optiml.ml.neural_network.initializers import (truncated_normal, glorot_normal,
                                                   glorot_uniform, he_normal, he_uniform)


def test_initializers_shape_finite_and_reproducible():
    shape = (10, 5)
    for init in (glorot_normal, glorot_uniform, he_normal, he_uniform):
        w = init(shape, random_state=42)
        assert w.shape == shape
        assert np.all(np.isfinite(w))
        # same seed must give the same weights
        assert np.allclose(w, init(shape, random_state=42))


def test_truncated_normal_is_bounded():
    std, mean = 1., 0.
    w = truncated_normal((10000,), mean=mean, std=std, random_state=0)
    assert np.all(np.abs(w - mean) <= 2 * std + 1e-9)


def test_uniform_initializers_within_limits():
    shape = (8, 4)
    glorot_limit = np.sqrt(6. / (shape[0] + shape[1]))
    assert np.all(np.abs(glorot_uniform(shape, random_state=0)) <= glorot_limit)
    he_limit = np.sqrt(6. / shape[0])
    assert np.all(np.abs(he_uniform(shape, random_state=0)) <= he_limit)


if __name__ == "__main__":
    pytest.main()
