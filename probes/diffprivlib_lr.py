"""
DiffPrivLib linear regression with Cebere et al. (2026) sensitivity fix.

Uses IBM's functional mechanism with corrected diagonal x^2 sensitivity
and PSD projection on the noisy Gram block.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

try:
    from diffprivlib.mechanisms import Laplace, LaplaceFolded
    from diffprivlib.utils import check_random_state
    import diffprivlib.models.linear_regression as _dplr_module

    _DPL_AVAILABLE = True
except ImportError:
    _DPL_AVAILABLE = False


def _project_psd(matrix: np.ndarray, eig_floor: float = 1e-6) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    w, v = np.linalg.eigh(matrix)
    w = np.maximum(w, eig_floor)
    return (v * w) @ v.T


def _construct_regression_obj_fixed(
    X: np.ndarray,
    y: np.ndarray,
    bounds_X: Tuple[np.ndarray, np.ndarray],
    bounds_y: Tuple[float, float],
    epsilon: float,
    alpha: float,
    random_state,
):
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_features = X.shape[1]
    n_targets = y.shape[1]

    local_epsilon = epsilon / (1 + n_targets * n_features + n_features * (n_features + 1) / 2)
    coefs = ((y ** 2).sum(axis=0), np.einsum("ij,ik->jk", X, y), np.einsum("ij,ik", X, X))

    def get_max_sensitivity(y_lower, y_upper, x_lower, x_upper):
        corners = [y_lower * x_lower, y_lower * x_upper, y_upper * x_lower, y_upper * x_upper]
        return np.max(corners) - np.min(corners)

    mono_coef_0 = np.zeros(n_targets)
    for i in range(n_targets):
        sensitivity = np.abs([bounds_y[0], bounds_y[1]]).max() ** 2
        mech = LaplaceFolded(
            epsilon=local_epsilon,
            sensitivity=sensitivity,
            lower=0,
            upper=float("inf"),
            random_state=random_state,
        )
        mono_coef_0[i] = mech.randomise(coefs[0][i])

    mono_coef_1 = np.zeros((n_features, n_targets))
    for i in range(n_targets):
        for j in range(n_features):
            sensitivity = get_max_sensitivity(
                bounds_y[0], bounds_y[1], bounds_X[0][j], bounds_X[1][j],
            )
            mech = Laplace(epsilon=local_epsilon, sensitivity=sensitivity, random_state=random_state)
            mono_coef_1[j, i] = mech.randomise(coefs[1][j, i])

    mono_coef_2 = np.zeros((n_features, n_features))
    for i in range(n_features):
        sensitivity = np.max(np.abs([bounds_X[0][i], bounds_X[1][i]])) ** 2
        mech = LaplaceFolded(
            epsilon=local_epsilon,
            sensitivity=sensitivity,
            lower=0,
            upper=float("inf"),
            random_state=random_state,
        )
        mono_coef_2[i, i] = mech.randomise(coefs[2][i, i])

        for j in range(i + 1, n_features):
            sensitivity = get_max_sensitivity(
                bounds_X[0][i], bounds_X[1][i], bounds_X[0][j], bounds_X[1][j],
            )
            mech = Laplace(epsilon=local_epsilon, sensitivity=sensitivity, random_state=random_state)
            mono_coef_2[i, j] = mech.randomise(coefs[2][i, j])
            mono_coef_2[j, i] = mono_coef_2[i, j]

    mono_coef_2 = _project_psd(mono_coef_2)
    noisy_coefs = (mono_coef_0, mono_coef_1, mono_coef_2)

    def obj(idx):
        def inner_obj(omega):
            func = noisy_coefs[0][idx]
            func -= 2 * np.dot(noisy_coefs[1][:, idx], omega)
            func += np.multiply(noisy_coefs[2], np.tensordot(omega, omega, axes=0)).sum()
            func += alpha * (omega ** 2).sum()
            grad = (
                -2 * noisy_coefs[1][:, idx]
                + 2 * np.matmul(noisy_coefs[2], omega)
                + 2 * omega * alpha
            )
            return func, grad

        return inner_obj

    return tuple(obj(i) for i in range(n_targets)), noisy_coefs


if _DPL_AVAILABLE:
    _dplr_module._construct_regression_obj = _construct_regression_obj_fixed


def _diffprivlib_fit(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    bounds_X: Tuple[np.ndarray, np.ndarray],
    bounds_y: Tuple[float, float],
    fit_intercept: bool = True,
    alpha: float = 1e-3,
    random_state=None,
):
    random_state = check_random_state(random_state)
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_features = X.shape[1]
    n_targets = y.shape[1]
    epsilon_intercept_scale = 1.0 / (n_features + 1) if fit_intercept else 0.0

    preprocess = _dplr_module._preprocess_data
    X, y, x_offset, y_offset, _x_scale = preprocess(
        X,
        y,
        fit_intercept=fit_intercept,
        bounds_X=bounds_X,
        bounds_y=bounds_y,
        epsilon=epsilon * epsilon_intercept_scale,
        copy=True,
        random_state=random_state,
    )

    bounds_X_c = (bounds_X[0] - x_offset, bounds_X[1] - x_offset)
    bounds_y_c = (bounds_y[0] - y_offset, bounds_y[1] - y_offset)

    objs, _ = _construct_regression_obj_fixed(
        X,
        y,
        bounds_X_c,
        bounds_y_c,
        epsilon=epsilon * (1.0 - epsilon_intercept_scale),
        alpha=alpha,
        random_state=random_state,
    )

    coef = np.zeros((n_features, n_targets))
    for i, obj in enumerate(objs):
        opt = minimize(obj, np.zeros(n_features), jac=True, method="L-BFGS-B")
        x = opt.x if np.isfinite(opt.x).all() else np.zeros(n_features)
        coef[:, i] = x

    coef = np.ravel(coef) if n_targets == 1 else coef.T
    intercept = float(np.ravel(y_offset - x_offset @ coef)[0])
    return intercept, coef


def diffprivlib_linear(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    bounds_X: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    bounds_y: Optional[Tuple[float, float]] = None,
    fit_intercept: bool = True,
) -> Optional[np.ndarray]:
    """
    DiffPrivLib functional-mechanism linear regression.
    Returns beta = [intercept, coef_1, ..., coef_p].
    """
    if not _DPL_AVAILABLE:
        return None

    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, p = X.shape

    if bounds_X is None:
        bounds_X_arr = (np.zeros(p), np.ones(p))
    else:
        lo = np.atleast_1d(np.asarray(bounds_X[0], float))
        hi = np.atleast_1d(np.asarray(bounds_X[1], float))
        bounds_X_arr = (
            np.repeat(lo, p) if lo.size == 1 else lo,
            np.repeat(hi, p) if hi.size == 1 else hi,
        )
    if bounds_y is None:
        bounds_y_arr = (0.0, 1.0)
    else:
        bounds_y_arr = (float(bounds_y[0]), float(bounds_y[1]))

    try:
        intercept, coef = _diffprivlib_fit(
            X,
            y,
            epsilon,
            bounds_X=bounds_X_arr,
            bounds_y=bounds_y_arr,
            fit_intercept=fit_intercept,
            alpha=1e-3,
        )
    except Exception:
        return None

    beta = np.concatenate([[intercept], np.ravel(coef)]) if fit_intercept else np.ravel(coef)
    return beta if np.isfinite(beta).all() else None


def diffprivlib_available() -> bool:
    return _DPL_AVAILABLE
