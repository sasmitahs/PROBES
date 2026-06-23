"""Unit-cube normalization and coefficient back-mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class AffineNorm:
    """Train-only min-max normalization for selected features."""

    x_min: np.ndarray
    x_max: np.ndarray
    y_min: float
    y_max: float
    feature_names: List[str]
    selected_idx: np.ndarray

    @property
    def x_delta(self) -> np.ndarray:
        return np.maximum(self.x_max - self.x_min, 1e-12)

    @property
    def y_delta(self) -> float:
        return max(self.y_max - self.y_min, 1e-12)

    def transform(self, X_sel: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_norm = (X_sel - self.x_min) / self.x_delta
        y_norm = (y - self.y_min) / self.y_delta
        return np.clip(x_norm, 0.0, 1.0), np.clip(y_norm, 0.0, 1.0)

    def inverse_coef(self, beta_norm: np.ndarray) -> Tuple[float, np.ndarray]:
        b0, slopes_n = float(beta_norm[0]), np.asarray(beta_norm[1:], float)
        slopes_orig = slopes_n * (self.y_delta / self.x_delta)
        intercept_orig = (
            self.y_min + self.y_delta * b0 - float(np.sum(slopes_orig * self.x_min))
        )
        return intercept_orig, slopes_orig

    def predict_orig(self, X_full: np.ndarray, beta_norm: np.ndarray) -> np.ndarray:
        intercept, slopes = self.inverse_coef(beta_norm)
        return intercept + X_full[:, self.selected_idx] @ slopes


def select_top_p_by_correlation(X: np.ndarray, y: np.ndarray, p: int) -> np.ndarray:
    yc = y - y.mean()
    ys = yc.std()
    if ys < 1e-12:
        ys = 1.0
    Xc = X - X.mean(axis=0)
    xs = Xc.std(axis=0)
    xs[xs < 1e-12] = 1.0
    corr = np.abs((Xc / xs).T @ yc / (len(y) * ys))
    p = min(p, X.shape[1])
    return np.argsort(corr)[::-1][:p]


def build_affine_norm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_idx: np.ndarray,
    feature_names: List[str] | None = None,
) -> AffineNorm:
    names = (
        [feature_names[i] for i in selected_idx]
        if feature_names is not None
        else [f"x{i}" for i in selected_idx]
    )
    xs = X_train[:, selected_idx]
    return AffineNorm(
        x_min=xs.min(axis=0),
        x_max=xs.max(axis=0),
        y_min=float(y_train.min()),
        y_max=float(y_train.max()),
        feature_names=names,
        selected_idx=np.asarray(selected_idx, int),
    )
