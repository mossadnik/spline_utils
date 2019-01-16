[![Build Status](https://travis-ci.org/mossadnik/spline_utils.svg?branch=master)](https://travis-ci.org/mossadnik/spline_utils)

# spline-utils - spline encoder and collection of functions for using splines for regression

This module contains a pure numpy implementation of B-splines, an sklearn compatible encoder, as well as utility functions for

 * setting knot positions based on deduplicated data quantiles
 * computing separate penalty matrices for curvature, linear and constant terms

While scipy already wraps a b-spline implementation, the module ships with its own implementation because

 * For spline regression one wants to compute all basis function values, which is not supported by scipy's splines and forces to use unasthetic for-loops
 * it enables changing e.g. extrapolation behavior (not done yet)
 * it's educating to do this

 The functions are consistent with scipy for the same knots, order, and derivative. Despite being pure python/numpy, on my laptop the performance is only around four times that of scipy for spline evaluation with a few basis functions. For encoding and/or many basis functions, performance is comparable or better than scipy (with more than around 10 basis functions for encoding, around 1000 for evaluation, see example folder).

# Installation

```
pip install git+https://github.com/mossadnik/spline_utils
```

# Basic usage

The simplest interface is the `SplineEncoder` (requires [scikit-learn](https://scikit-learn.org) to be installed):

```python
import numpy as np
from spline_utils.encoder import SplineEncoder

x = np.arange(50, dtype=float)
encoder = SplineEncoder(knots=4).fit(x)
encoder.transform(np.array([0]))

array([[1., 0., 0., 0., 0., 0.]])
```

There is optional handling of missing values. These are added as an additional column, so that rows still sum to one:

```python
x[1] = np.nan
encoder = SplineEncoder(knots=4, nullable=True).fit(x)
encoder.transform(x[:2])
array([[1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1.]])
```

For penalized regression / smoothing splines you can create a curvature penalty matrix as well as projectors onto the linear and constant parts (see examples folder):

```python
import spline_utils as spl

penalty = spl.get_cubic_spline_penalty(encoder.knots_, return_nullspace=False)
```

# Up next

 * add extrapolation, e.g. constant, linear, or closest polynomial piece
 * handling special values for mixed categorical/numerical features
 * more freedom in defining knots for the `SplineEncoder` 
