import numpy as np


def skew(m):
    """Compute the skew-symmetric matrix for m"""
    m = np.asarray(m)
    assert m.shape == (3,), 'skew() received shape %s' % str(m.shape)
    return np.array([[0, -m[2], m[1]],
                     [m[2], 0, -m[0]],
                     [-m[1], m[0], 0.]])


def exp(m):
    """Compute the mapping from so(3) to SO(3)"""
    m = np.asarray(m)
    assert np.shape(m) == (3,), 'exp() received shape %s' % str(m.shape)

    tsq = np.dot(m, m)
    if tsq < 1e-8:
        # Taylor expansion of sin(sqrt(x))/sqrt(x):
        #   http://www.wolframalpha.com/input/?i=sin(sqrt(x))/sqrt(x)
        a = 1. - tsq/6. + tsq*tsq/120.

        # Taylor expansion of (1 - cos(sqrt(x))/x:
        #   http://www.wolframalpha.com/input/?i=(1-cos(sqrt(x)))/x
        b = .5 - tsq/24. + tsq*tsq/720.
    else:
        t = np.sqrt(tsq)
        a = np.sin(t)/t
        b = (1. - np.cos(t)) / tsq

    sk = skew(m)
    return np.eye(3) + a*sk + b*np.dot(sk, sk)


def exp_jacobian(x):
    """Compute the jacobian of exp(x) w.r.t. x. More specifically, this function return a 3x3 matrix A such that
    for infinitesimal 3-vectors delta exp(A * delta) * exp(x) = exp(x + delta)"""
    tsq = np.dot(x, x)
    if tsq < 1e-8:
        # Taylor expansion:
        # http://www.wolframalpha.com/input/?i=2+*+sin%28x%2F2%29+*+sin%28x%2F2%29+%2F+%28x*x%29
        a = .5 - tsq/24 + tsq*tsq/720
        # Taylor expansion:
        # http://www.wolframalpha.com/input/?i=%28x-sin%28x%29%29%2Fx%5E3
        b = 1./6. - tsq/120 + tsq*tsq/5040
    else:
        t = np.sqrt(tsq)
        a = -2. * np.sin(t/2.) * np.sin(t/2.) / tsq
        b = (t - np.sin(t)) / (t*t*t)

    sk = skew(x)
    return np.transpose(np.eye(3) + a * sk + b * np.dot(sk, sk))


def log(r):
    """Compute the mapping from SO(3) to so(3)"""
    r = np.asarray(r)
    assert np.shape(r) == (3, 3), 'log() received shape %s' % str(np.shape(r))

    # http://math.stackexchange.com/questions/83874/
    t = float(r.trace())
    x = np.array((r[2, 1] - r[1, 2],
                  r[0, 2] - r[2, 0],
                  r[1, 0] - r[0, 1]))
    if t >= 3. - 1e-8:
        return (.5 - (t-3.)/12.) * x
    elif t > -1. + 1e-8:
        th = np.arccos(t/2. - .5)
        return th / (2. * np.sin(th)) * x
    else:
        assert t <= -1. + 1e-8, 't=%f, R=%s' % (t, r)
        a = int(np.argmax(r[np.diag_indices_from(r)]))
        b = (a+1) % 3
        c = (a+2) % 3
        s = np.sqrt(r[a,a] - r[b, b] - r[c, c] + 1.)
        v = np.empty(3)
        v[a] = s/2.
        v[b] = (r[b, a] + r[a, b]) / (2.*s)
        v[c] = (r[c, a] + r[a, c]) / (2.*s)
        return v / np.linalg.norm(v)
