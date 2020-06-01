import numpy as np


def zeros(shape, random_state=None):
    return np.zeros(shape)


def ones(shape, random_state=None):
    return np.ones(shape)


def constant(v, shape, random_state=None):
    return np.full(shape, v)


def random_normal(shape, mean=0., std=1., random_state=None):
    return np.random.RandomState(random_state).normal(loc=mean, scale=std, size=shape)


def random_uniform(shape, low=0., high=1., random_state=None):
    return np.random.RandomState(random_state).uniform(low=low, high=high, size=shape)


def truncated_normal(shape, mean=0., std=1., random_state=None):
    truncated = 2 * std + mean
    return np.clip(random_normal(shape=shape, mean=mean, std=std,
                                 random_state=random_state), -truncated, truncated)


def glorot_normal(shape, channels_last=True, random_state=None):
    """Glorot normal initializer, also called Xavier normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with std = sqrt(2 / (fan_in + fan_out))
    where fan_in is the number of input units in the weight tensor
    and fan_out is the number of output units in the weight tensor."""
    fan_in, fan_out = compute_fans(shape, channels_last)
    std = np.sqrt(2. / (fan_in + fan_out))
    return truncated_normal(shape=(fan_in, fan_out), mean=0., std=std, random_state=random_state)


def glorot_uniform(shape, channels_last=True, random_state=None):
    """Glorot uniform initializer, also called Xavier uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where limit is sqrt(6 / (fan_in + fan_out))
    where fan_in is the number of input units in the weight tensor
    and fan_out is the number of output units in the weight tensor."""
    fan_in, fan_out = compute_fans(shape, channels_last)
    limit = np.sqrt(6. / (fan_in + fan_out))
    return random_uniform(shape=shape, low=-limit, high=limit, random_state=random_state)


def he_normal(shape, channels_last=True, random_state=None):
    """He normal initializer.cIt draws samples from a truncated normal
    distribution centered on 0 with std = sqrt(2 / fan_in) where
    fan_in is the number of input units in the weight tensor."""
    fan_in, fan_out = compute_fans(shape, channels_last)
    std = np.sqrt(2. / fan_in)
    return truncated_normal(shape=shape, mean=0., std=std, random_state=random_state)


def he_uniform(shape, channels_last=True, random_state=None):
    """He uniform variance scaling initializer. It draws samples from
    a uniform distribution within [-limit, limit] where limit is
    sqrt(6 / fan_in) where fan_in is the number of input units in
    the weight tensor."""
    fan_in, fan_out = compute_fans(shape, channels_last)
    limit = np.sqrt(6. / fan_in)
    return random_uniform(shape=shape, low=-limit, high=limit, random_state=random_state)


def compute_fans(shape, channels_last=True):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:  # Conv2D
        if channels_last:
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:  # channels_first
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
    return fan_in, fan_out
