import numpy as np


def _recursion(x, bins, knots, order, deriv, res):
    """B-spline recursion formula"""
    for k in range(1, order + 1):
        if k <= order - deriv:  # standard recursion
            for i in range(order - k, order + 1):
                if i > order - k:  # B[i, k - 1] -> B[i, k]
                    tk, ti = knots[bins + i + k], knots[bins + i]
                    res[i] *= (x - ti) / (tk - ti)
                if i < order:  # B[i + 1, k - 1] -> B[i, k]
                    tk, ti = knots[bins + i + k + 1], knots[bins + i + 1]
                    res[i] += (tk - x) / (tk - ti) * res[i + 1]
        else:  # derivative recursion
            for i in range(order - k, order + 1):
                if i > order - k:
                    tk, ti = knots[bins + i + k], knots[bins + i]
                    res[i] *= k / (tk - ti)
                if i < order:
                    tk, ti = knots[bins + i + k + 1], knots[bins + i + 1]
                    res[i] += -k / (tk - ti) * res[i + 1]


def get_spline_values(x, knots, order=3, deriv=0):
    """Evaluate b-spline basis functions.

    This is used internally for both spline evaluation
    and encoding for regression.

    implements standard B-spline recursion formulas, see e.g.
    https://en.wikipedia.org/wiki/B-spline

    Parameters
    ----------
    x : array
        data, one-dimensional, must be contained in interval
        defined by knots
    knots : array
        knot locations
    order : int
        order of the spline, default to cubic
    deriv : int
        order of the derivative, defaults to no derivative

    Returns
    -------
    tuple
        containing arrays (basis-function indices, basis-function values)
        for all data values x non-vanishing basis-functions
    """
    if np.any((x > knots.max()) | (x < knots.min())):
        raise ValueError("some x are out of bounds (%f, %f)" % (knots.min(), knots.max()))
    # b contains bin assignments for initializing recursion
    b = np.atleast_1d(np.searchsorted(knots[order:knots.size - order], x, side='right') - 1)
    # extrapolation, currently not used, instead ValueError above
    b[b < 0] = 0
    b[b == knots.size - 2 * order - 1] = knots.size - 2 * order - 2
    if x.ndim == 0:
        b = b[0]

    # store (order + 1) non-vanishing coefficients per value
    res = np.zeros((order + 1, ) + x.shape)
    res[-1] = 1.

    _recursion(x, b, knots, order, deriv, res)

    if res.ndim > 1:
        b = b[..., None] + np.arange(order + 1).reshape((1,) * b.ndim + (-1,))
        res = np.transpose(res, np.roll(np.arange(res.ndim), -1))
    else:
        b = b + np.arange(order + 1)
    return b, res


def spline_eval(x, knots, coef, order=3, deriv=0):
    """evaluate B-spline

    Parameters
    ----------
    x : array
        data, must be contained in interval defined by
        knot locations
    knots : array
        knot locations
    coef : array
        coefficients of the basis functions
    order : int
       order of the spline, defaults to cubic
    deriv : int
       order of derivative, defaults to no derivative

    Returns
    -------
    array
        function values
    """
    ix, values = get_spline_values(x, knots, order, deriv)
    return np.sum(coef[ix] * values, axis=-1)


def spline_encode(x, knots, order=3, deriv=0):
    """get B-spline basis function values.

    Parameters
    ----------
    x : array
        data, must be one-dimensional and free of nans
    knots : array
        knot locations
    order : int
       order of the spline, defaults to cubic
    deriv : int
       order of derivative, defaults to no derivative

    Returns
    -------
    array
        basis function values, shape (x.size, knots.size - (order + 1))
    """
    x = np.asarray(x)
    if not x.ndim == 1:
        raise ValueError('only one-dimensional input supported')

    ix, values = get_spline_values(x, knots, order, deriv)
    res = np.zeros((x.size, knots.size - (order + 1)))
    res[np.arange(x.size)[:, None], ix] = values
    return res
