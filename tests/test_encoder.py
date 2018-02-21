import unittest
import numpy as np
from scipy.interpolate import splev
import spline_utils as spl
from spline_utils.encoder import SplineEncoder

class TestSplineEncoderSplev(unittest.TestCase):
    def test_spline_encoder_splev(self):
        """test that spline encoder is consistent with splev"""
        x = np.arange(100, dtype=float)
        n_knots = 10
        for order in range(4):
            knots = spl.quantile_knots(x, n_knots, order=order)
            # get basis function coefficients from splev
            s = knots.size - (order + 1)
            expected = np.empty((x.size, s))
            coef = np.zeros(s)
            for i in range(s):
                coef[i] = 1
                expected[:, i] = splev(x, (knots, coef, order))
                coef[i] = 0
            # get same result from SplineEncoder
            actual = SplineEncoder(knots=n_knots, order=order, nullable=False).fit_transform(x)
            self.assertTrue(np.allclose(expected, actual), "unexpected result for order={}".format(order))


class TestNullable(unittest.TestCase):
    def test_catch_notnullable_nans(self):
        """setting nullable to False raises exception when encountering np.nan"""
        with self.assertRaises(ValueError):
            SplineEncoder(nullable=False).fit_transform(np.array([1., np.nan]))

    def test_nullable(self):
        """check that nulls are handled as intended (adding one column, populating last column for na)"""
        x = np.arange(11, dtype=float)
        x[[1, -1]] = np.nan
        null = np.isnan(x)

        kw = dict(knots=5)
        res_null = SplineEncoder(nullable=True, **kw).fit_transform(x)
        res_notnull = SplineEncoder(nullable=False, **kw).fit_transform(x[~null])
        self.assertEquals(res_null.shape[1], 1 + res_notnull.shape[1], 'result shape inconsistent')
        self.assertTrue(np.allclose(res_null[~null, :-1], res_notnull), 'non na-cases do not match non-nullable')
        self.assertTrue(np.allclose(res_null[null, -1], 1.), 'na-value not set')
