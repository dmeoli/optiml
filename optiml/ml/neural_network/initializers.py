import numpy as np


def truncated_normal(shape, mean=0., std=1., random_state=None):
    truncated = 2 * std + mean
    return np.clip(np.random.RandomState(random_state).normal(size=shape, loc=mean, scale=std), -truncated, truncated)


def glorot_normal(shape, random_state=None):
    """Glorot normal initializer, also called Xavier normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with std = sqrt(2 / (fan_in + fan_out))
    where fan_in is the number of input units in the weight tensor
    and fan_out is the number of output units in the weight tensor."""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2. / (fan_in + fan_out))
    return truncated_normal(shape=shape, mean=0., std=std, random_state=random_state)


def glorot_uniform(shape, random_state=None):
    """Glorot uniform initializer, also called Xavier uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where limit is sqrt(6 / (fan_in + fan_out))
    where fan_in is the number of input units in the weight tensor
    and fan_out is the number of output units in the weight tensor."""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6. / (fan_in + fan_out))
    return np.random.RandomState(random_state).uniform(size=shape, low=-limit, high=limit)


def he_normal(shape, random_state=None):
    """He normal initializer.cIt draws samples from a truncated normal
    distribution centered on 0 with std = sqrt(2 / fan_in) where
    fan_in is the number of input units in the weight tensor."""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2. / fan_in)
    return truncated_normal(shape=shape, mean=0., std=std, random_state=random_state)


def he_uniform(shape, random_state=None):
    """He uniform variance scaling initializer. It draws samples from
    a uniform distribution within [-limit, limit] where limit is
    sqrt(6 / fan_in) where fan_in is the number of input units in
    the weight tensor."""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6. / fan_in)
    return np.random.RandomState(random_state).uniform(size=shape, low=-limit, high=limit)
