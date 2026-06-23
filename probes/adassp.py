"""AdaSSP: adaptive sufficient-statistic perturbation (Wang 2018)."""

from __future__ import annotations

from math import log
from typing import Optional

import numpy as np


def prepare_adassp_design(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Build AdaSSP design Z = [1 | X] and row bound BZ = sqrt(p+1).
    X should satisfy coordinate-wise bounds in [0, 1].
    """
    X = np.asarray(X, float)
    n, p = X.shape
    design = np.column_stack([np.ones(n), X])
    return design, float(np.sqrt(p + 1.0))


def adassp(
    Z: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    delta: Optional[float] = None,
    BZ: Optional[float] = None,
    BY: float = 1.0,
    rho: float = 0.05,
    ZTZ_pre: Optional[np.ndarray] = None,
    ZTy_pre: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    AdaSSP Algorithm 2. Returns beta = [intercept, slopes...].
    """
    Z = np.asarray(Z, float)
    y = np.asarray(y, float).ravel()
    n, d = Z.shape

    if BZ is None:
        BZ = float(np.sqrt(d))
    if delta is None:
        delta = min(1e-6, 1.0 / (n ** 2))

    eps_split = epsilon / 3.0
    lsd = log(6.0 / delta)

    eta = np.sqrt(d * lsd * log(2.0 * d ** 2 / rho)) * (BZ ** 2) / eps_split

    zty = ZTy_pre if ZTy_pre is not None else Z.T @ y
    ztz = ZTZ_pre if ZTZ_pre is not None else Z.T @ Z

    lambda_min_exact = float(np.linalg.eigvalsh(ztz)[0])
    noise_std_eig = np.sqrt(lsd) * (BZ ** 2) / eps_split
    lm_shift = lsd * (BZ ** 2) / eps_split
    lambda_min_private = max(
        lambda_min_exact + noise_std_eig * np.random.randn() - lm_shift,
        0.0,
    )
    lam = max(0.0, eta - lambda_min_private)

    noise_std_cross = np.sqrt(lsd) * BZ * BY / eps_split
    zty_hat = zty + noise_std_cross * np.random.randn(d)

    w = np.zeros((d, d))
    triu_i, triu_j = np.triu_indices(d)
    w[triu_i, triu_j] = np.random.randn(len(triu_i))
    w = w + w.T - np.diag(np.diag(w))
    noise_std_gram = np.sqrt(lsd) * (BZ ** 2) / eps_split
    ztz_hat = ztz + noise_std_gram * w

    try:
        theta = np.linalg.solve(ztz_hat + lam * np.eye(d), zty_hat)
        return theta if np.isfinite(theta).all() else None
    except np.linalg.LinAlgError:
        return None


def adassp_linear(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    delta: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Convenience wrapper on raw feature matrix."""
    Z, BZ = prepare_adassp_design(X)
    return adassp(Z, y, epsilon, delta=delta, BZ=BZ, BY=1.0)
