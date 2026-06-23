"""Bernstein basis utilities shared by PROBES and Ada-PROBES."""

from __future__ import annotations

import numpy as np
from scipy.special import comb


def bernstein_basis_1d(x: np.ndarray, order: int = 2) -> np.ndarray:
    """Evaluate order-`order` Bernstein basis at each x_i. Shape (n, order+1)."""
    x = np.asarray(x, float)
    n = x.shape[0]
    basis = np.zeros((n, order + 1))
    for k in range(order + 1):
        basis[:, k] = comb(order, k) * (x ** k) * ((1.0 - x) ** (order - k))
    return basis


def bernstein_to_monomial_weights(order: int = 2) -> np.ndarray:
    """
    Lemma 2.1 weights: x^m = sum_k w[m,k] p_k(x),  w[m,k] = C(k,m)/C(order,m).
    """
    weights = np.zeros((order + 1, order + 1))
    for m in range(order + 1):
        for k in range(m, order + 1):
            weights[m, k] = comb(k, m) / comb(order, m)
    return weights


def tensor_cells(X: np.ndarray, order: int = 2) -> np.ndarray:
    """Tensor-product Bernstein cells for X in [0,1]^(n,p). Shape (n, (order+1)^p)."""
    n, p = X.shape
    cells = np.ones((n, 1))
    for j in range(p):
        basis_j = bernstein_basis_1d(X[:, j], order)
        cells = (cells[:, :, None] * basis_j[:, None, :]).reshape(n, -1)
    return cells


def tensor_weight_vector(alpha: np.ndarray, weights_1d: np.ndarray) -> np.ndarray:
    """Kronecker product of W1d[alpha_j] across predictors."""
    w = np.array([1.0])
    for a in alpha:
        w = np.kron(w, weights_1d[a])
    return w
