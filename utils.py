import numpy as np


def normalized(x):
    x = np.asarray(x)
    return x / np.sqrt(np.sum(np.square(x), axis=-1))[..., None]


def pr(x):
    x = np.asarray(x)
    return x[..., :-1] / x[..., -1:]


def unpr(x):
    x = np.asarray(x)
    col_shape = x.shape[:-1] + (1,)
    return np.concatenate((x, np.ones(col_shape)), axis=-1)


def spy(x, tol=1e-4):
    x = np.atleast_2d(x)
    return '\n'.join(map(lambda row: '['+''.join('x' if abs(val)>tol else ' ' for val in row)+']', x))


def unreduce(x, mask, fill=0.):
    x = np.asarray(x)
    out = np.repeat(fill, len(mask))
    out[mask] = x
    return out


def unreduce_info(info, mask):
    out = np.zeros((len(mask), len(mask)))
    out[np.ix_(mask, mask)] = info
    return out


def cis(theta):
    """This works for both scalar and vector theta."""
    return np.array((np.cos(theta), np.sin(theta)))


def dots(*m):
    """Multiple an arbitrary number of matrices with np.dot."""
    return reduce(np.dot, m)


def sumsq(x, axis=None):
    """Compute the sum of squared elements."""
    return np.sum(np.square(x), axis=axis)


def skew(m):
    """Compute the skew-symmetric matrix for m."""
    m = np.asarray(m)
    assert m.shape == (3,), 'shape was was %s' % str(m.shape)
    return np.array([[0, -m[2], m[1]],
                     [m[2], 0, -m[0]],
                     [-m[1], m[0], 0.]])


def unit(i, n):
    return (np.arange(n) == i).astype(float)


def orthonormalize(r):
    u, s, v = np.linalg.svd(r)
    return np.dot(u, v)
