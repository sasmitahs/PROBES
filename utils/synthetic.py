"""Synthetic data generators for benchmark experiments."""

from __future__ import annotations

import numpy as np

TRAIN_FRAC = 0.8


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = TRAIN_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sequential 80/20 train/test split.

    Matches ``exp_linear_d1.py``: first ``train_frac`` of rows for training,
    remainder held out for test MSE on ``y``.
    """
    n = len(y)
    n_train = int(train_frac * n)
    if n_train <= 0 or n_train >= n:
        raise ValueError(f"invalid split: n={n}, train_frac={train_frac}")
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def generate_iid_linear(
    n: int,
    p: int,
    intercept: float = 0.2,
    noise_sd: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    i.i.d. Uniform features with clipped linear response in [0,1].

    Returns X, y, true_beta where true_beta = [intercept, slopes...].
    """
    rng = np.random.RandomState(seed)
    slopes = np.full(p, 0.6 / p)
    X_raw = rng.uniform(0, 1, (n, p))
    y_raw = intercept + X_raw @ slopes + noise_sd * rng.randn(n)
    X = np.clip(X_raw, 0.0, 1.0)
    y = np.clip(y_raw, 0.0, 1.0)
    return X, y, np.concatenate([[intercept], slopes])


def generate_iid_linear_split(
    n: int,
    p: int,
    train_frac: float = TRAIN_FRAC,
    intercept: float = 0.2,
    noise_sd: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate i.i.d. data with a sequential 80/20 train/test split."""
    X, y, _ = generate_iid_linear(n, p, intercept=intercept, noise_sd=noise_sd, seed=seed)
    return split_train_test(X, y, train_frac=train_frac)


def generate_correlated_linear(
    n: int,
    p: int,
    intercept: float = 0.2,
    noise_sd: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Correlated block + i.i.d. block features.
    """
    rng = np.random.RandomState(seed)
    n_corr = p // 2
    n_iid = p - n_corr

    if n_corr > 0:
        z = rng.randn(n, n_corr)
        a = rng.randn(n_corr, n_corr)
        corr_raw = z @ a.T
        corr_min = corr_raw.min(axis=0)
        corr_max = corr_raw.max(axis=0)
        corr_span = np.maximum(corr_max - corr_min, 1e-12)
        X_corr = (corr_raw - corr_min) / corr_span
    else:
        X_corr = np.empty((n, 0))

    if n_iid > 0:
        X_iid = rng.uniform(0, 1, (n, n_iid))
    else:
        X_iid = np.empty((n, 0))

    X_raw = np.column_stack([X_corr, X_iid]) if p > 0 else np.empty((n, 0))
    slopes = np.full(p, 0.6 / p)
    y_raw = intercept + X_raw @ slopes + noise_sd * rng.randn(n)
    X = np.clip(X_raw, 0.0, 1.0)
    y = np.clip(y_raw, 0.0, 1.0)
    return X, y, np.concatenate([[intercept], slopes])


def generate_correlated_linear_split(
    n: int,
    p: int,
    train_frac: float = TRAIN_FRAC,
    intercept: float = 0.2,
    noise_sd: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate correlated-block data with sequential train/test split."""
    X, y, _ = generate_correlated_linear(
        n, p, intercept=intercept, noise_sd=noise_sd, seed=seed,
    )
    return split_train_test(X, y, train_frac=train_frac)
