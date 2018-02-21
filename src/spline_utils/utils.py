import numpy as np
import scipy.sparse as sparse
from .core import spline_encode


def add_boundary_knots(knots, order=3, a=None, b=None):
    """add boundary knots to array of interior knots.

    output has correct format for use in spline functions.

    Parameters
    ----------
    knots : sequence of float
        sorted knot values
    order : int
        order of the spline
    a : int, optional
        left boundary value, defaults to first knot
    b : int, optional
        right boundary value, defaults to last knot

    Returns
    -------
    array
        knots with added boundary values, shape is
        (knots.size + 2 * order + (a is not None) + (b is not None))
    """
    na, nb = [order + 1] * 2
    knots = np.asarray(knots)
    if a is None or a == knots[0]:
        a = knots[0]
        na -= 1
    if b is None or b == knots[-1]:
        b = knots[-1]
        nb -= 1
    if a > knots[0]:
        raise ValueError('left boundary a inside interval: %f > %f' % (a, knots[0]))
    if b < knots[-1]:
        raise ValueError('right boundary b inside interval: %f < %f' % (b, knots[-1]))
    shape = knots.shape[1:]
    return np.concatenate((a * np.ones((na,) + shape), knots, b * np.ones((nb,) + shape)))


def quantile_knots(x, n=8, order=3, a=None, b=None):
    """compute knot positions from empirical quantiles.

    Parameters
    ----------
    x : array
        data
    n : int
        number of interior knots. Additional boundary knots are added
        depending on the order of the spline
    a : float
       left interval boundary, set to min(x) if None
    b : float
       upper interval boundary, set to max(x) if None

    Returns
    -------
    array
        knot locations, including 2 * (order + 1) boundary knots
    """
    xu = np.unique(x)
    if xu.size < n:
        n = xu.size
    n = n - 2
    if n < 1:
        raise ValueError('too few values, min(n, x.size) < 3')
    q = 100. * np.arange(1, n + 1) / (n + 1)
    res = np.percentile(xu, q)
    a = a or xu[0]
    b = b or xu[-1]
    return add_boundary_knots(res, order=order, a=a, b=b)


def get_cubic_spline_penalty(knots, return_null_space=False):
    """Get penalty matrices for cubic splines.

    see e.g. https://arxiv.org/abs/0707.0143

    Computes matrix representation of the usual curvature penalty as well as (optionally)
    projections to the constant and linear components that span the null-space of the curvature
    penalty.

    Parameters
    ----------
    knots : array
        knots of the spline

    return_null_space : bool
        whether to return null-space projectors

    Returns
    -------
    matrices : array or tuple of arrays
        if return_null_space is False, returns the
        curvature penalty matrix.
        if return_null_space is True, returns a tuple
        (curvature_penalty, null_space), where null_space
        has shape (knots.size, 2) and contains projection
        vectors for the constant and linear terms.
    """
    x = np.zeros((3, knots.size - 1))
    w = np.zeros_like(x)

    # positions
    x[0] = knots[:-1]
    x[2] = knots[1:]
    x[1] = x[[0, 2]].mean(axis=0)
    x = x.T.ravel()

    # integration weights
    w[:] = np.diff(knots)[None, :]
    w[[0, 2]] /= 6.
    w[1] *= 2. / 3.
    w = sparse.diags(w.T.ravel())

    # matrix
    B = sparse.csr_matrix(spline_encode(x, knots, deriv=2))
    res = .5 * (B.T * w * B).toarray()
    # add small offset to ensure positivity
    res += 1e-8 * np.eye(res.shape[0])

    if not return_null_space:
        return res

    # orthogonal nullspace weight vectors for const / linear term
    nullspace = np.ones((2, res.shape[0]))
    k = 3
    # linear apart from offset
    w = np.cumsum(knots[k:-1] - knots[:-k-1]) / k
    # subtract projection on constant vector
    w -= w.mean()
    nullspace[1] = w
    nullspace = nullspace / np.sqrt(np.sum(nullspace**2, axis=1, keepdims=True))

    return res, nullspace
