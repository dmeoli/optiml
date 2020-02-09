import numpy as np


class Initializer:
    def initialize(self, x):
        raise NotImplementedError


class RandomNormal(Initializer):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self.mean, scale=self.std, size=x.shape)


class RandomUniform(Initializer):
    def __init__(self, low=0., high=1.):
        self.low = low
        self.high = high

    def initialize(self, x):
        x[:] = np.random.uniform(self.low, self.high, size=x.shape)


class Zeros(Initializer):
    def initialize(self, x):
        x[:] = np.zeros_like(x)


class Ones(Initializer):
    def initialize(self, x):
        x[:] = np.ones_like(x)


class TruncatedNormal(Initializer):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def initialize(self, x):
        truncated = 2 * self.std + self.mean
        x[:] = np.clip(np.random.normal(loc=self.mean, scale=self.std, size=x.shape), -truncated, truncated)


class Constant(Initializer):
    def __init__(self, v):
        self.v = v

    def initialize(self, x):
        x[:] = np.full_like(x, self.v)


class GlorotNormal(Initializer):
    """Glorot normal initializer, also called Xavier normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    """

    def initialize(self, x):
        TruncatedNormal(mean=0., std=np.sqrt(2. / (x.shape[0] + x.shape[1]))).initialize(x)


class GlorotUniform(Initializer):
    """Glorot uniform initializer, also called Xavier uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    """

    def initialize(self, x):
        limit = np.sqrt(6. / (x.shape[0] + x.shape[1]))
        RandomUniform(low=limit, high=-limit).initialize(x)
