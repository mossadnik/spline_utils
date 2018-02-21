import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .core import spline_encode
from .utils import add_boundary_knots, quantile_knots


class SplineEncoder(BaseEstimator, TransformerMixin):
    """Encode numeric features into B-spline basis functions"""
    def __init__(self, knots=10, order=3, interval=None, nullable=False):
        """
        Parameters
        ----------
        knots : int
            number of unique knots for the spline (discounting duplicated boundary knots).
            Note that there are more basis functions than interior knots, depending on the
            order. The number of columns returned is knots + order - 1.
        order : int
            order of the spline. order = 0 corresponds to binning, order = 1 to
            piecewise linear etc. The first order derivatives are smooth.
        interval : tuple, optional
            lower and upper boundary of interval, must cover the fitted data if specified.
            If None (default), use min and max of data as interval.
        nullable : bool
            If true, an additional column is added to the update that is one for
            np.nan. This preserves the property that the sum over features is unity
            for each observation.
            If false (default), an ValueError is raised when trying to process data containing
            np.nan
        """

        self.n_knots = knots
        self._knots = None
        self.order = order
        self.interval = interval
        self.nullable = nullable

    def fit(self, x, y=None):
        """compute knot values from quantiles of deduplicated data

        Parameters
        ----------
        x : array
            must be one-dimensional
        y : None
            dummy for sklearn compatibility
        """
        x = np.asarray(x)
        a, b = None, None
        if self.interval is not None:
            a, b = self.interval
        null = self._check_nan(x)

        self._knots = quantile_knots(x[~null], self.n_knots, self.order, a, b)
        self.n_columns_ = self._knots.size - (self.order + 1)
        if self.nullable:
            self.n_columns_ += 1
        return self

    def _check_fitted(self):
        if self._knots is None:
            raise RuntimeError('SplineEncoder needs to be fitted first')

    def _check_nan(self, x):
        null = np.isnan(x)
        if not self.nullable and np.max(null):
            raise ValueError('Input contains nan, but SplineEncoder.nullable is set to False')
        return null

    @property
    def knots_(self):
        self._check_fitted()
        return self._knots

    def transform(self, x, y=None):
        """apply spline encoding to data

        Parameters
        ----------
        x : array
            must be one-dimensional
        y : None
            dummy for sklearn compatibility

        Returns
        -------
        array
            encoded values, shape (x.size, knots + order - 1 + nullable),
            rows sum to one
        """
        self._check_fitted()
        null = self._check_nan(x)
        notnull = np.where(~null)[0]
        res = np.zeros((x.shape[0], self.n_columns_))
        nc = self.n_columns_ if not self.nullable else self.n_columns_ - 1
        res[notnull, :nc] = spline_encode(x[notnull], self._knots, self.order)
        if self.nullable:
            res[null, -1] = 1.

        return res

    def fit_transform(self, x, y=None):
        """fit and apply spline encoding to data

        Parameters
        ----------
        x : array
            must be one-dimensional
        y : None
            dummy for sklearn compatibility

        Returns
        -------
        array
            encoded values, shape (x.size, knots + order - 1 + nullable),
            rows sum to one
        """
        return self.fit(x).transform(x)
