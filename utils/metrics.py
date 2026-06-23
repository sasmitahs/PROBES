"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, float)
    X = np.asarray(X, float)
    return beta[0] + X @ beta[1:]


def coefficient_mse(beta_hat: np.ndarray, beta_true: np.ndarray) -> float:
    if beta_hat is None or not np.isfinite(beta_hat).all():
        return float("nan")
    return float(np.mean((beta_hat - beta_true) ** 2))
