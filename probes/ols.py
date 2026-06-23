"""Non-private ordinary least squares baseline."""

from __future__ import annotations

import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit OLS with intercept.

    Returns beta = [intercept, slope_1, ..., slope_p].
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    design = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(design, y, rcond=None)[0]
