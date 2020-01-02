"""
This module holds various schedules for parameters such as the step
rate or momentum for gradient descent.

A schedule is implemented as an iterator. This allows it to have iterators
of infinite length. It also makes it possible to manipulate scheduls with
the ``itertools`` python module, e.g. for chaining iterators.
"""

import itertools

import numpy as np


def decaying(start, decay):
    """
    Return an iterator of exponentially decaying values.
    The first value is ``start``. Every further value is obtained by multiplying
    the last one by a factor of ``decay``.
    """
    return (start * decay ** i for i in itertools.count(0))


def linear_annealing(start, stop, n_steps):
    """
    Return an iterator that anneals linearly to a point linearly.
    The first value is ``start``, the last value is ``stop``. The annealing will
    be linear over ``n_steps`` iterations. After that, ``stop`` is yielded.
    """
    start, stop = float(start), float(stop)
    inc = (stop - start) / n_steps
    for i in range(n_steps):
        yield start + i * inc
    while True:
        yield stop


def repeater(iter, n):
    """
    Return an iterator that repeats each element of `iter` exactly
    `n` times before moving on to the next element.
    """
    for i in iter:
        for j in range(n):
            yield i


class SutskeverBlend:
    """
    Class representing a schedule that step-wise increases from zero to a
    maximum value, as described in [sutskever2013importance]
    On the importance of initialization and momentum in deep learning,
    Sutskever et al (ICML 2013)
    """

    def __init__(self, max_momentum, stretch=250):
        self.max_momentum = max_momentum
        self.stretch = stretch

    def __iter__(self):
        for i in itertools.count(1):
            m = 1 - (2 ** (-1 - np.log2(np.floor_divide(i, self.stretch) + 1)))
            yield min(m, self.max_momentum)
