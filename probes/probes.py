"""
PROBES: single-shot Bernstein simplex release (Algorithm 1).

Releases all Bernstein cells in one Laplace draw at scale 1/epsilon
(ell_1-sensitivity = 1). Sufficient statistics are recovered by
linear post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from probes.bernstein import (
    bernstein_to_monomial_weights,
    tensor_cells,
    tensor_weight_vector,
)


@dataclass
class ProbesRelease:
    """Noisy Bernstein cell sums after the single Laplace release."""

    tA: np.ndarray  # noisy B_{k,y} cells, shape (K,)
    tB: np.ndarray  # noisy B_{k,ybar} cells, shape (K,)
    p: int
    order: int


class ProbesCache:
    """
    Precomputed Bernstein cell sums for a fixed training set.

    The joint vector [A_c; B_c] has size 2 * 3^p and ell_1-sensitivity 1.
    One Laplace(1/epsilon) draw gives pure epsilon-DP.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, p = X.shape
        self.n, self.p = n, p
        order = 2  # Bernstein order for linear regression (d=1)
        self.order = order

        cells = tensor_cells(X, order)
        self.K = cells.shape[1]

        self.A_c = cells.T @ y
        self.B_c = cells.T @ (1.0 - y)
        self.N_joint = 2 * self.K

        weights_1d = bernstein_to_monomial_weights(order)
        self.weights_1d = weights_1d
        dim = p + 1
        self.D = dim

        def feat_alpha(a: int, b: int) -> np.ndarray:
            alpha = np.zeros(p, dtype=int)
            if a >= 1:
                alpha[a - 1] += 1
            if b >= 1:
                alpha[b - 1] += 1
            return alpha

        self._zz_pairs: List[Tuple[int, int, np.ndarray]] = []
        self._zy_entries: List[Tuple[int, np.ndarray]] = []
        for a in range(dim):
            for b in range(a, dim):
                w = tensor_weight_vector(feat_alpha(a, b), weights_1d)
                self._zz_pairs.append((a, b, w))
            self._zy_entries.append((a, tensor_weight_vector(feat_alpha(a, 0), weights_1d)))

    def release(self, epsilon: float, rng: np.random.Generator | None = None) -> ProbesRelease:
        """Single-shot Laplace release at full budget epsilon."""
        if rng is None:
            joint_noise = np.random.laplace(0.0, 1.0 / epsilon, size=self.N_joint)
        else:
            joint_noise = rng.laplace(0.0, 1.0 / epsilon, size=self.N_joint)
        tA = self.A_c + joint_noise[: self.K]
        tB = self.B_c + joint_noise[self.K :]
        return ProbesRelease(tA=tA, tB=tB, p=self.p, order=self.order)

    def assemble_gram(self, release: ProbesRelease) -> Tuple[np.ndarray, np.ndarray]:
        """Recover noisy Gram matrix and cross-product from released cells."""
        tN = release.tA + release.tB
        dim = self.D
        gram = np.zeros((dim, dim))
        cross = np.zeros(dim)
        for a, b, w in self._zz_pairs:
            v = float(w @ tN)
            gram[a, b] = gram[b, a] = v
        for a, w in self._zy_entries:
            cross[a] = float(w @ release.tA)
        return gram, cross


def probes_release(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> ProbesRelease:
    """Run PROBES release on bounded data in [0,1]^(p+1)."""
    cache = ProbesCache(X, y)
    return cache.release(epsilon, rng=rng)


def probes_recover_statistics(
    release: ProbesRelease,
    cache: ProbesCache,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-process a PROBES release into (Gram, cross-product)."""
    return cache.assemble_gram(release)
