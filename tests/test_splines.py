import numpy as np
from scipy.interpolate import splev
import spline_utils as spl


def test_spline_eval():
    """check that spline evaluation reproduces results
    of scipy.interpolate.splev for different orders
    and derivatives.
    """
    np.random.seed(1234)
    n = 1000
    n_knot = 20

    x = np.random.randn(n)
    for order in range(4):
        knots = spl.quantile_knots(x, 20, order=order)
        coef = np.random.rand(knots.size - (order + 1))
        for deriv in range(order):
            expected = splev(x, (knots, coef, order), der=deriv)
            actual = spl.spline_eval(x, knots, coef, order=order, deriv=deriv)
            assert np.allclose(actual, expected)


def test_add_boundary_knots():
    """number of boundary knots added is correct"""
    knots = np.array([-1, 0, 1])
    n = knots.size
    a, b = -2, 2
    for order in range(4):
        assert spl.add_boundary_knots(knots, order, a=a, b=b).size == 2 * (order + 1) + n
        assert spl.add_boundary_knots(knots, order, a=None, b=b).size == 2 * (order + 1) + n - 1
        assert spl.add_boundary_knots(knots, order, a=a, b=None).size == 2 * (order + 1) + n - 1
        assert spl.add_boundary_knots(knots, order, a=None, b=None).size == 2 * order + n
