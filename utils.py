import itertools
import random


def not_test(func):
    """Decorator to mark a function or method as *not* a test"""
    func.__test__ = False
    return func


def arbitrary_slice(arr, start, stop=None, axis=0):
    """Return a slice from `start` to `stop` in dimension `axis` of array `arr`.


    Parameters
    ----------

    arr : array_like
        Can be numpy ndarray, hdf5 dataset, or list.
        If ``arr`` is a list, ``axis`` must be 0.

    start : int
        Index at which to start slicing.

    stop : int, optional [default: None]
        Index at which to stop slicing.
        If not specified, the axis is sliced until its end.

    axis : int, optional [default: 0]
        Axis along which should be sliced


    Returns
    -------

    slice : array_like
        The respective slice of ``arr``
    """

    if type(arr) is list:
        if axis == 0:
            return arr[start:stop]
        else:
            raise ValueError("Cannot slice a list in non-zero axis {}".format(axis))

    n_axes = len(arr.shape)

    if axis >= n_axes:
        raise IndexError('Argument `axis` with value {} out of range. '
                         'Must be smaller than rank {} of `arr`.'.format(axis, n_axes))

    this_slice = [slice(None) for _ in range(n_axes)]
    this_slice[axis] = slice(start, stop)

    return arr[tuple(this_slice)]


def iter_mini_batches(lst, batch_size, dims, n_cycles=None, random_state=None, discard_illsized_batch=False):
    """Return an iterator that successively yields tuples containing aligned
    mini batches of size `batch_size` from sliceable objects given in `lst`, in
    random order without replacement.
    Because different containers might require slicing over different
    dimensions, the dimension of each container has to be givens as a list
    `dims`.


    Parameters
    ----------

    lst : list of array_like
        Each item of the list will be sliced into mini batches in alignment
        with the others.

    batch_size : int
        Size of each batch. Last batch might be smaller.

    dims : list
        Aligned with ``lst``, gives the dimension along which the data samples
        are separated.

    n_cycles : int, optional [default: None]
        Number of cycles after which to stop the iterator. If ``None``, will
        yield forever.

    random_state : a numpy.random.RandomState object, optional [default : None]
        Random number generator that will act as a seed for the mini batch order.

    discard_illsized_batch : bool, optional [default : False]
        If ``True`` and the length of the sliced dimension is not divisible by
        ``batch_size``, the leftover samples are discarded.


    Returns
    -------
    batches : iterator
        Infinite iterator of mini batches in random order (without replacement).
    """

    # This if clause is for backward compatibility.
    if type(n_cycles) == bool and not n_cycles:
        n_cycles = None
        warnings.warn("n_cycles=False kwarg to iter_mini_batches deprecated. "
                      "Using n_cycles=None instead.")

    try:
        # case distinction for handling lists
        dm_result = [divmod(len(arr), batch_size)
                     if d == 0 else divmod(arr.shape[d], batch_size)
                     for (arr, d) in zip(lst, dims)]
    except AttributeError:
        raise AttributeError("'list' object has no attribute 'shape'. "
                             "Trying to slice a list in a non-zero axis.")
    except IndexError:
        raise IndexError("tuple index out of range. "
                         "Trying to slice along a non-existing dimension.")

    # check if all to-be-sliced dimensions have the same length
    if dm_result.count(dm_result[0]) == len(dm_result):
        n_batches, rest = dm_result[0]
        if rest and not discard_illsized_batch:
            n_batches += 1
    else:
        raise ValueError("The axes along which to slice have unequal lengths.")

    if random_state is not None:
        random.seed(random_state.normal())

    counter = itertools.count()
    count = next(counter)

    while True:
        indices = range(n_batches)
        while True:
            if n_cycles is not None and count >= n_cycles:
                raise StopIteration()
            count = next(counter)

            random.shuffle(indices)
            for i in indices:
                start = i * batch_size
                stop = (i + 1) * batch_size
                batch = [arbitrary_slice(arr, start, stop, axis) for (arr, axis)
                         in zip(lst, dims)]
                yield tuple(batch)


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
    """The argument is a string; convert to a number if possible, or strip it."""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()
