import numpy as np


def not_test(func):
    """Decorator to mark a function or method as *not* a test"""
    func.__test__ = False
    return func


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


def isnumber(x):
    """Is x a number?"""
    return hasattr(x, '__int__')


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
