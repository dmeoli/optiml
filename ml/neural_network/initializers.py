import numpy as np


def zeros(shape):
    return np.zeros(shape)


def ones(shape):
    return np.ones(shape)


def constant(shape, value):
    return np.full(shape, value)


def random_normal(shape, mean=0, stddev=0.05):
    return np.random.normal(mean, stddev, shape)


def random_uniform(shape, low=-0.05, high=0.05):
    return np.random.uniform(low, high, shape)


class VarianceScaling:
    """Initializer capable of adapting its scale to the shape of weights.
    With `distribution="normal"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:
        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"
    With `distribution="uniform"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
    # Arguments
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "normal", "uniform".
        seed: A Python integer. Used to seed the random generator.
    # Raises
        ValueError: In case of an invalid value for the "scale", mode" or
          "distribution" arguments.
    """

    def __init__(self, scale=1.0, mode='fan_in', distribution='normal', seed=None):
        if scale <= 0.:
            raise ValueError('`scale` must be a positive float. Got:', scale)
        mode = mode.lower()
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument: '
                             'expected on of {"fan_in", "fan_out", "fan_avg"} '
                             'but got', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype=None):
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self.distribution == 'normal':
            # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(scale) / .8796256610342398
            x = K.truncated_normal(shape, 0., stddev, dtype=dtype, seed=self.seed)
        else:
            limit = np.sqrt(3. * scale)
            x = K.random_uniform(shape, -limit, limit, dtype=dtype, seed=self.seed)
        if self.seed is not None:
            self.seed += 1
        return x


def glorot_normal(seed=None):
    """Glorot normal initializer, also called Xavier normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    return VarianceScaling(scale=1., mode='fan_avg', distribution='normal', seed=seed)


def glorot_uniform(seed=None):
    """Glorot uniform initializer, also called Xavier uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    return VarianceScaling(scale=1., mode='fan_avg', distribution='uniform', seed=seed)
