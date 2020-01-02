import collections
import os
import bisect
import random
from statistics import mean

import numpy as np

from optimization_test_functions import Function


# loss functions

def cross_entropy_loss(x, y):
    """Example of cross entropy loss. x and y are 1D iterable objects."""
    return (-1.0 / len(x)) * sum(x * np.log(y) + (1 - x) * np.log(1 - y) for x, y in zip(x, y))


def mse_loss(x, y):
    """Example of min square loss. x and y are 1D iterable objects."""
    return (1.0 / len(x)) * sum((_x - _y) ** 2 for _x, _y in zip(x, y))


# activation functions

def clip(x, lowest, highest):
    """Return x clipped to the range [lowest..highest]."""
    return max(lowest, min(x, highest))


def softmax1D(x):
    """Return the softmax vector of input vector x."""
    exps = [np.exp(_x) for _x in x]
    sum_exps = sum(exps)
    return [exp / sum_exps for exp in exps]


def conv1D(x, k):
    """1D convolution. x: input vector; K: kernel vector."""
    return np.convolve(x, k, mode='same')


class Sigmoid(Function):

    def function(self, x):
        if x >= 100:
            return 1
        if x <= -100:
            return 0
        return 1 / (1 + np.exp(-x))

    def jacobian(self, x):
        return x * (1 - x)


class Relu(Function):

    def function(self, x):
        return max(0, x)

    def jacobian(self, x):
        return 1 if x > 0 else 0


class Elu(Function):

    def function(self, x, alpha=0.01):
        return x if x > 0 else alpha * (np.exp(x) - 1)

    def jacobian(self, x, alpha=0.01):
        return 1 if x > 0 else alpha * np.exp(x)


class Tanh(Function):

    def function(self, x):
        return np.tanh(x)

    def jacobian(self, x):
        return 1 - (x ** 2)


class LeakyRelu(Function):

    def function(self, x, alpha=0.01):
        return x if x > 0 else alpha * x

    def jacobian(self, x, alpha=0.01):
        return 1 if x > 0 else alpha


def random_weights(min_value, max_value, num_weights):
    return [random.uniform(min_value, max_value) for _ in range(num_weights)]


# kernels

def gaussian(mean, st_dev, x):
    """Given the mean and standard deviation of a distribution, it returns the probability of x."""
    return 1 / (np.sqrt(2 * np.pi) * st_dev) * np.exp(-0.5 * (float(x - mean) / st_dev) ** 2)


def gaussian_kernel(size=3):
    return [gaussian((size - 1) / 2, 0.1, x) for x in range(size)]


def linear_kernel(x, y=None):
    if y is None:
        y = x
    return np.dot(x, y.T)


def polynomial_kernel(x, y=None, degree=2.0):
    if y is None:
        y = x
    return (1.0 + np.dot(x, y.T)) ** degree


def rbf_kernel(x, y=None, gamma=None):
    """Radial-basis function kernel (aka squared-exponential kernel)."""
    if y is None:
        y = x
    if gamma is None:
        gamma = 1.0 / x.shape[1]  # 1.0 / n_features
    return np.exp(-gamma * (-2.0 * np.dot(x, y.T) +
                            np.sum(x * x, axis=1).reshape((-1, 1)) + np.sum(y * y, axis=1).reshape((1, -1))))


def open_data(name, mode='r'):
    return open(os.path.join(os.path.dirname(__file__), *['data', name]), mode=mode)


def remove_all(item, seq):
    """Return a copy of seq (or string) with all occurrences of item removed."""
    if isinstance(seq, str):
        return seq.replace(item, '')
    elif isinstance(seq, set):
        rest = seq.copy()
        rest.remove(item)
        return rest
    else:
        return [x for x in seq if x != item]


def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements."""
    return list(set(seq))


def num_or_str(x):
    """The argument is a string; convert to a number if
       possible, or strip it."""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()


def mean_boolean_error(x, y):
    return mean(_x != _y for _x, _y in zip(x, y))


def normalize(dist):
    """Multiply each number by a constant such that the sum is 1.0"""
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1  # probabilities must be between 0 and 1
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]


def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)


def weighted_sample_with_replacement(n, seq, weights):
    """Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight."""
    sample = weighted_sampler(seq, weights)

    return [sample() for _ in range(n)]


def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


# ______________________________________________________________________________
# argmin and argmax

identity = lambda x: x


def argmin_random_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random."""
    return min(shuffled(seq), key=key)


def argmax_random_tie(seq, key=identity):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return max(shuffled(seq), key=key)


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


def isnumber(x):
    """Is x a number?"""
    return hasattr(x, '__int__')


def issequence(x):
    """Is x a sequence?"""
    return isinstance(x, collections.abc.Sequence)


def print_table(table, header=None, sep='   ', numfmt='{}'):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '{:.2f}'.
    (If you want different formats in different columns,
    don't use print_table.) sep is the separator between columns."""
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]

    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row]
             for row in table]

    sizes = list(map(lambda seq: max(map(len, seq)), list(zip(*[map(str, row) for row in table]))))

    for row in table:
        print(sep.join(getattr(str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))


def scalar_vector_product(x, y):
    """Return vector as a product of a scalar and a vector recursively."""
    return [scalar_vector_product(x, _y) for _y in y] if hasattr(y, '__iter__') else x * y


def map_vector(f, x):
    """Apply function f to iterable x."""
    return [map_vector(f, _x) for _x in x] if hasattr(x, '__iter__') else list(map(f, [x]))[0]


def dot_product(x, y):
    """Return the sum of the element-wise product of vectors x and y."""
    return sum(_x * _y for _x, _y in zip(x, y))


def element_wise_product(x, y):
    if hasattr(x, '__iter__') and hasattr(y, '__iter__'):
        assert len(x) == len(y)
        return [element_wise_product(_x, _y) for _x, _y in zip(x, y)]
    elif hasattr(x, '__iter__') == hasattr(y, '__iter__'):
        return x * y
    else:
        raise Exception('Inputs must be in the same size!')


def matrix_multiplication(x, *y):
    """Return a matrix as a matrix-multiplication of x and arbitrary number of matrices *y."""

    result = x
    for _y in y:
        result = np.matmul(result, _y)

    return result


def vector_add(a, b):
    """Component-wise addition of two vectors."""
    if not (a and b):
        return a or b
    if hasattr(a, '__iter__') and hasattr(b, '__iter__'):
        assert len(a) == len(b)
        return list(map(vector_add, a, b))
    else:
        try:
            return a + b
        except TypeError:
            raise Exception('Inputs must be in the same size!')
