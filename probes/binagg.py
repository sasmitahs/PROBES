"""BinAgg: binned aggregation DP linear regression (Lin et al. 2025)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

try:
    from binagg import dp_linear_regression

    _BINAGG_AVAILABLE = True
except ImportError:
    _BINAGG_AVAILABLE = False


def delta_from_gdp(mu: float, eps: float) -> float:
    """delta(epsilon) for a mu-GDP mechanism."""
    if mu <= 0 or eps < 0:
        raise ValueError("mu must be positive and eps non-negative")
    return float(max(
        0.0,
        norm.cdf(-eps / mu + mu / 2.0)
        - np.exp(eps) * norm.cdf(-eps / mu - mu / 2.0),
    ))


def mu_from_eps_delta(eps: float, delta: float, mu_range: tuple[float, float] = (1e-5, 100.0)) -> float:
    """Find mu so mu-GDP implies (epsilon, delta)-DP."""
    if eps <= 0 or not (0.0 < delta < 1.0):
        raise ValueError("need eps > 0 and 0 < delta < 1")
    return float(brentq(lambda mu: delta_from_gdp(mu, eps) - delta, *mu_range))


def binagg_linear(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    delta: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    BinAgg private linear regression on [0,1]-bounded data.
    Returns beta = [intercept, coef_1, ..., coef_p].
    """
    if not _BINAGG_AVAILABLE:
        return None

    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, p = X.shape
    if delta is None:
        delta = min(1e-6, 1.0 / (n ** 2))

    try:
        mu = mu_from_eps_delta(epsilon, delta)
    except (ValueError, RuntimeError):
        return None

    ones_col = np.ones(n) + np.random.uniform(-1e-4, 1e-4, n)
    try:
        res = dp_linear_regression(
            np.column_stack([ones_col, X]),
            y,
            x_bounds=[(0.999, 1.001)] + [(0.0, 1.0)] * p,
            y_bounds=(0.0, 1.0),
            mu=mu,
        )
        coef = res.coefficients
        return np.concatenate([[float(coef[0])], np.ravel(coef[1:])])
    except Exception:
        return None


def binagg_available() -> bool:
    return _BINAGG_AVAILABLE
